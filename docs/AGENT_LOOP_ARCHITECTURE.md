# Claude Code Agent Loop 完整实现文档

> 本文档基于 claude-code 2.1.88 源码逆向分析，目标是让另一个 agent 可以完整复刻出 Agent Loop 的核心逻辑。

## 1. 核心文件清单

| 文件 | 职责 |
|------|------|
| `src/query.ts` | **Agent Loop 主体** — 包含 `query()` 和 `queryLoop()` |
| `src/QueryEngine.ts` | 上层编排器，持有会话状态，调用 `query()` |
| `src/query/deps.ts` | 依赖注入（callModel, microcompact, autocompact, uuid） |
| `src/query/config.ts` | 不可变配置快照（循环入口时冻结一次） |
| `src/query/stopHooks.ts` | 停止钩子评估（决定模型完成后是否需要继续） |
| `src/query/tokenBudget.ts` | Token 预算续写逻辑 |
| `src/services/api/claude.ts` | `queryModelWithStreaming()` — 流式 API 调用与响应解析 |
| `src/services/api/withRetry.ts` | 重试逻辑（指数退避） |
| `src/services/tools/toolOrchestration.ts` | 工具执行编排（串行 vs 并发） |
| `src/services/tools/toolExecution.ts` | 单工具执行（`runToolUse`） |
| `src/services/tools/StreamingToolExecutor.ts` | 流式工具执行（模型流式输出期间就开始执行工具） |
| `src/services/compact/autoCompact.ts` | Auto-compact 触发逻辑和阈值计算 |

---

## 2. 整体架构概览

```
用户输入
    │
    ▼
QueryEngine.ask()  ─── 构建 systemPrompt / 处理附件 / 管理会话
    │
    ▼
query()  ─── 外壳函数，管理 command lifecycle
    │
    ▼
queryLoop()  ─── 核心 while(true) 循环
    │
    ├─→ 消息预处理（budget/snip/microcompact/collapse）
    ├─→ autoCompact 检查与执行
    ├─→ 调用 LLM API（流式）
    ├─→ 流式工具执行（边流边执行）
    ├─→ 停止条件判断 / 错误恢复
    ├─→ 收集附件 / 内存预取
    └─→ 组装新 state，continue 到下一轮
```

---

## 3. 核心数据结构

### 3.1 循环状态 State

```typescript
type State = {
  messages: Message[]                    // 完整对话历史
  toolUseContext: ToolUseContext          // 工具执行上下文（包含 tools 列表、abortController 等）
  autoCompactTracking: AutoCompactTrackingState | undefined
  maxOutputTokensRecoveryCount: number   // max_output_tokens 恢复计数（上限 3）
  hasAttemptedReactiveCompact: boolean   // 是否已尝试过响应式压缩
  maxOutputTokensOverride: number | undefined  // 输出 token 限制覆盖
  pendingToolUseSummary: Promise<ToolUseSummaryMessage | null> | undefined
  stopHookActive: boolean | undefined    // stop hook 是否激活
  turnCount: number                      // 当前轮次号
  transition: Continue | undefined       // 上一轮为何继续（用于调试/断言）
}
```

### 3.2 查询参数 QueryParams

```typescript
type QueryParams = {
  messages: Message[]                          // 初始消息列表
  systemPrompt: SystemPrompt                   // 系统提示词（string[]）
  userContext: { [k: string]: string }         // 用户上下文（注入到每轮消息前）
  systemContext: { [k: string]: string }       // 系统上下文（追加到 systemPrompt 后）
  canUseTool: CanUseToolFn                     // 工具权限判断函数
  toolUseContext: ToolUseContext               // 工具使用上下文
  fallbackModel?: string                       // 降级模型
  querySource: QuerySource                     // 来源标识（'repl_main_thread' | 'agent:*' | 'sdk' 等）
  maxOutputTokensOverride?: number             // 输出 token 覆盖
  maxTurns?: number                            // 最大轮次限制
  skipCacheWrite?: boolean                     // 跳过缓存写入
  taskBudget?: { total: number }               // 任务预算
  deps?: QueryDeps                             // 依赖注入（测试用）
}
```

### 3.3 依赖注入 QueryDeps

```typescript
type QueryDeps = {
  callModel: typeof queryModelWithStreaming   // 调用模型的函数
  microcompact: typeof microcompactMessages  // 微压缩
  autocompact: typeof autoCompactIfNeeded    // 自动压缩
  uuid: () => string                         // UUID 生成
}

function productionDeps(): QueryDeps {
  return {
    callModel: queryModelWithStreaming,
    microcompact: microcompactMessages,
    autocompact: autoCompactIfNeeded,
    uuid: randomUUID,
  }
}
```

### 3.4 消息类型 Message

```typescript
// 助手消息
type AssistantMessage = {
  type: 'assistant'
  message: {
    content: ContentBlock[]      // text | tool_use | thinking | redacted_thinking
    usage: Usage
    stop_reason: StopReason | null
    model: string
  }
  uuid: string
  timestamp: string
  requestId?: string
  isApiErrorMessage?: boolean
  apiError?: 'max_output_tokens' | 'invalid_request'
}

// 用户消息（也用于 tool_result）
type UserMessage = {
  type: 'user'
  message: {
    content: ContentBlockParam[]  // text | tool_result | image
  }
  uuid: string
  timestamp: string
  toolUseResult?: string
  sourceToolAssistantUUID?: string
  isMeta?: boolean               // 系统注入的消息（不是用户真正输入的）
}

// 附件消息
type AttachmentMessage = {
  type: 'attachment'
  attachment: { type: string; ... }
}
```

### 3.5 工具调用结果格式

```typescript
// 工具结果作为 UserMessage 回传
createUserMessage({
  content: [
    {
      type: 'tool_result',
      content: resultString,        // 工具执行结果文本
      is_error: boolean,            // 是否出错
      tool_use_id: toolUse.id,      // 关联的 tool_use block ID
    }
  ],
  toolUseResult: resultString,
  sourceToolAssistantUUID: assistantMessage.uuid,
})
```

---

## 4. 核心循环详细流程

### 4.1 入口 `query()`

```typescript
async function* query(params: QueryParams): AsyncGenerator<...> {
  const consumedCommandUuids: string[] = []
  const terminal = yield* queryLoop(params, consumedCommandUuids)
  // 通知已消费的命令完成
  for (const uuid of consumedCommandUuids) {
    notifyCommandLifecycle(uuid, 'completed')
  }
  return terminal
}
```

### 4.2 主循环 `queryLoop()` — 完整伪代码

```typescript
async function* queryLoop(params, consumedCommandUuids) {
  // === 初始化 ===
  const { systemPrompt, userContext, systemContext, canUseTool, fallbackModel, maxTurns } = params
  const deps = params.deps ?? productionDeps()
  
  let state: State = {
    messages: params.messages,
    toolUseContext: params.toolUseContext,
    maxOutputTokensOverride: params.maxOutputTokensOverride,
    autoCompactTracking: undefined,
    stopHookActive: undefined,
    maxOutputTokensRecoveryCount: 0,
    hasAttemptedReactiveCompact: false,
    turnCount: 1,
    pendingToolUseSummary: undefined,
    transition: undefined,
  }
  
  const budgetTracker = createBudgetTracker()
  const config = buildQueryConfig()  // 冻结一次配置
  
  // 启动内存预取（异步，后续消费）
  using pendingMemoryPrefetch = startRelevantMemoryPrefetch(state.messages, state.toolUseContext)

  // === 主循环 ===
  while (true) {
    let { toolUseContext } = state
    const { messages, autoCompactTracking, turnCount, ... } = state

    yield { type: 'stream_request_start' }

    // ============ STEP 1: 消息预处理 ============
    let messagesForQuery = [...getMessagesAfterCompactBoundary(messages)]
    
    // 1a. 工具结果预算裁剪
    messagesForQuery = await applyToolResultBudget(messagesForQuery, ...)
    
    // 1b. 历史裁剪 (snip)
    if (HISTORY_SNIP) {
      const snipResult = snipCompactIfNeeded(messagesForQuery)
      messagesForQuery = snipResult.messages
    }
    
    // 1c. 微压缩（移除旧的 tool_result）
    messagesForQuery = await deps.microcompact(messagesForQuery, toolUseContext, querySource)
    
    // 1d. 上下文折叠
    if (CONTEXT_COLLAPSE) {
      messagesForQuery = (await contextCollapse.applyCollapsesIfNeeded(...)).messages
    }

    // ============ STEP 2: 自动压缩 ============
    const { compactionResult, consecutiveFailures } = await deps.autocompact(
      messagesForQuery, toolUseContext, cacheSafeParams, querySource, tracking, snipTokensFreed
    )
    
    if (compactionResult) {
      // 压缩成功：用压缩后的消息替换
      messagesForQuery = buildPostCompactMessages(compactionResult)
      for (const msg of messagesForQuery) { yield msg }
      tracking = { compacted: true, turnId: uuid(), turnCounter: 0, consecutiveFailures: 0 }
    }

    // ============ STEP 3: 阻塞限制检查 ============
    if (!compactionResult && !reactiveCompactEnabled) {
      const { isAtBlockingLimit } = calculateTokenWarningState(tokenCount, model)
      if (isAtBlockingLimit) {
        yield createAssistantAPIErrorMessage({ content: PROMPT_TOO_LONG_ERROR_MESSAGE })
        return { reason: 'blocking_limit' }
      }
    }

    // ============ STEP 4: 调用模型（流式） ============
    const assistantMessages: AssistantMessage[] = []
    const toolResults: (UserMessage | AttachmentMessage)[] = []
    const toolUseBlocks: ToolUseBlock[] = []
    let needsFollowUp = false
    
    let streamingToolExecutor = new StreamingToolExecutor(tools, canUseTool, toolUseContext)
    let currentModel = getRuntimeMainLoopModel(...)
    let attemptWithFallback = true

    while (attemptWithFallback) {
      attemptWithFallback = false
      try {
        for await (const message of deps.callModel({
          messages: prependUserContext(messagesForQuery, userContext),
          systemPrompt: fullSystemPrompt,
          thinkingConfig,
          tools: toolUseContext.options.tools,
          signal: toolUseContext.abortController.signal,
          options: { model: currentModel, fallbackModel, maxOutputTokensOverride, ... }
        })) {
          // 处理流式降级
          if (streamingFallbackOccured) {
            // 清空状态，重新开始
            assistantMessages.length = 0; toolResults.length = 0; toolUseBlocks.length = 0
            needsFollowUp = false
            streamingToolExecutor.discard()
            streamingToolExecutor = new StreamingToolExecutor(...)
          }
          
          // 扣留可恢复错误（prompt-too-long, max-output-tokens）
          let withheld = false
          if (isPromptTooLong(message) || isMaxOutputTokens(message)) { withheld = true }
          if (!withheld) { yield message }
          
          // 收集助手消息和工具调用
          if (message.type === 'assistant') {
            assistantMessages.push(message)
            const blocks = extractToolUseBlocks(message)
            if (blocks.length > 0) {
              toolUseBlocks.push(...blocks)
              needsFollowUp = true
            }
            // 立即开始流式工具执行
            for (const block of blocks) {
              streamingToolExecutor.addTool(block, message)
            }
          }
          
          // 收集已完成的流式工具结果
          for (const result of streamingToolExecutor.getCompletedResults()) {
            yield result.message
            toolResults.push(normalizeForAPI(result.message))
          }
        }
      } catch (innerError) {
        if (innerError instanceof FallbackTriggeredError && fallbackModel) {
          // 切换到降级模型并重试
          currentModel = fallbackModel
          attemptWithFallback = true
          continue
        }
        throw innerError
      }
    }

    // ============ STEP 5: 中断检查 ============
    if (signal.aborted) {
      yield createUserInterruptionMessage({ toolUse: false })
      return { reason: 'aborted_streaming' }
    }

    // ============ STEP 6: 停止条件判断（无 tool_use 时） ============
    if (!needsFollowUp) {
      const lastMessage = assistantMessages.at(-1)
      
      // 6a. prompt-too-long 恢复
      if (isWithheld413) {
        // 先尝试 context collapse drain
        if (contextCollapse.recoverFromOverflow(...).committed > 0) {
          state = { ...state, transition: { reason: 'collapse_drain_retry' } }
          continue
        }
        // 再尝试 reactive compact
        if (await reactiveCompact.tryReactiveCompact(...)) {
          state = { ...state, hasAttemptedReactiveCompact: true, transition: { reason: 'reactive_compact_retry' } }
          continue
        }
        // 恢复失败，报错退出
        yield lastMessage
        return { reason: 'prompt_too_long' }
      }
      
      // 6b. max_output_tokens 恢复
      if (isWithheldMaxOutputTokens(lastMessage)) {
        // 第一步：升级到 64k token 限制
        if (maxOutputTokensOverride === undefined) {
          state = { ...state, maxOutputTokensOverride: ESCALATED_MAX_TOKENS, transition: { reason: 'max_output_tokens_escalate' } }
          continue
        }
        // 第二步：注入 "resume" 消息（最多 3 次）
        if (maxOutputTokensRecoveryCount < 3) {
          const recoveryMessage = createUserMessage({
            content: 'Output token limit hit. Resume directly — no apology, no recap...',
            isMeta: true,
          })
          state = { 
            ...state, 
            messages: [...messagesForQuery, ...assistantMessages, recoveryMessage],
            maxOutputTokensRecoveryCount: count + 1,
            transition: { reason: 'max_output_tokens_recovery' }
          }
          continue
        }
        // 恢复耗尽，显示错误
        yield lastMessage
      }
      
      // 6c. API 错误消息直接退出
      if (lastMessage?.isApiErrorMessage) {
        return { reason: 'completed' }
      }
      
      // 6d. 运行 stop hooks
      const stopHookResult = yield* handleStopHooks(...)
      if (stopHookResult.preventContinuation) {
        return { reason: 'stop_hook_prevented' }
      }
      if (stopHookResult.blockingErrors.length > 0) {
        // hook 返回了阻塞错误，需要模型处理
        state = {
          ...state,
          messages: [...messagesForQuery, ...assistantMessages, ...stopHookResult.blockingErrors],
          stopHookActive: true,
          transition: { reason: 'stop_hook_blocking' },
        }
        continue
      }
      
      // 6e. Token 预算续写检查
      const decision = checkTokenBudget(budgetTracker, agentId, budget, turnOutputTokens)
      if (decision.action === 'continue') {
        state = {
          ...state,
          messages: [...messagesForQuery, ...assistantMessages, createUserMessage({ content: decision.nudgeMessage, isMeta: true })],
          transition: { reason: 'token_budget_continuation' },
        }
        continue
      }
      
      // === 正常退出 ===
      return { reason: 'completed' }
    }

    // ============ STEP 7: 执行工具 ============
    const toolUpdates = streamingToolExecutor
      ? streamingToolExecutor.getRemainingResults()
      : runTools(toolUseBlocks, assistantMessages, canUseTool, toolUseContext)

    for await (const update of toolUpdates) {
      if (update.message) {
        yield update.message
        toolResults.push(normalizeForAPI(update.message))
      }
      if (update.newContext) {
        updatedToolUseContext = update.newContext
      }
    }

    // ============ STEP 8: 工具执行后中断检查 ============
    if (signal.aborted) {
      yield createUserInterruptionMessage({ toolUse: true })
      return { reason: 'aborted_tools' }
    }
    if (shouldPreventContinuation) {
      return { reason: 'hook_stopped' }
    }

    // ============ STEP 9: 收集附件 ============
    // 包括：文件变更附件、队列命令、内存预取结果、技能发现结果
    for await (const attachment of getAttachmentMessages(...)) {
      yield attachment
      toolResults.push(attachment)
    }
    
    // 消费内存预取
    if (pendingMemoryPrefetch?.settledAt !== null) {
      const memoryAttachments = filterDuplicateMemoryAttachments(await pendingMemoryPrefetch.promise, ...)
      for (const att of memoryAttachments) { yield att; toolResults.push(att) }
    }
    
    // 刷新 MCP 工具（动态连接的 MCP 服务器）
    if (updatedToolUseContext.options.refreshTools) {
      updatedToolUseContext.options.tools = updatedToolUseContext.options.refreshTools()
    }

    // ============ STEP 10: 最大轮次检查 ============
    const nextTurnCount = turnCount + 1
    if (maxTurns && nextTurnCount > maxTurns) {
      yield createAttachmentMessage({ type: 'max_turns_reached', maxTurns, turnCount: nextTurnCount })
      return { reason: 'max_turns', turnCount: nextTurnCount }
    }

    // ============ STEP 11: 循环继续 ============
    state = {
      messages: [...messagesForQuery, ...assistantMessages, ...toolResults],
      toolUseContext: updatedToolUseContext,
      autoCompactTracking: tracking,
      turnCount: nextTurnCount,
      maxOutputTokensRecoveryCount: 0,
      hasAttemptedReactiveCompact: false,
      pendingToolUseSummary: nextPendingToolUseSummary,
      maxOutputTokensOverride: undefined,
      stopHookActive,
      transition: { reason: 'next_turn' },
    }
  } // while (true) END
}
```

---

## 5. 停止条件汇总

| 返回值 reason | 触发条件 |
|---|---|
| `completed` | 模型响应中没有 `tool_use` 块（正常结束） |
| `blocking_limit` | Token 数超过阻塞限制（auto-compact 禁用时） |
| `prompt_too_long` | API 返回 prompt-too-long 且所有恢复手段失败 |
| `image_error` | 图片大小/媒体错误不可恢复 |
| `model_error` | API 调用抛出未处理异常 |
| `aborted_streaming` | 用户在模型流式输出期间中断 |
| `aborted_tools` | 用户在工具执行期间中断 |
| `hook_stopped` | 工具 hook 阻止了继续 |
| `stop_hook_prevented` | Stop hooks 显式阻止继续 |
| `max_turns` | 轮次超过 `maxTurns` 限制 |

## 6. 继续条件汇总（不退出循环）

| transition reason | 触发条件 |
|---|---|
| `next_turn` | 有 tool_use 块，执行完工具后继续 |
| `max_output_tokens_escalate` | 从默认 8k 升级到 64k 输出限制重试 |
| `max_output_tokens_recovery` | 注入 "resume" 消息继续（最多 3 次） |
| `reactive_compact_retry` | 响应式压缩成功后重试 |
| `collapse_drain_retry` | Context collapse 释放了空间后重试 |
| `stop_hook_blocking` | Stop hook 返回阻塞错误，模型需要处理 |
| `token_budget_continuation` | Token 预算未耗尽，nudge 模型继续 |

---

## 7. 流式 API 调用详解

### 7.1 `queryModelWithStreaming()` (`services/api/claude.ts`)

核心流程：

```typescript
async function* queryModel(messages, systemPrompt, thinkingConfig, tools, signal, options) {
  // 1. 构建工具 schema
  const toolSchemas = await Promise.all(filteredTools.map(tool => toolToAPISchema(tool, ...)))
  
  // 2. 规范化消息
  let messagesForAPI = normalizeMessagesForAPI(messages, filteredTools)
  
  // 3. 构建系统提示词
  const system = buildSystemPromptBlocks(systemPrompt, enablePromptCaching, ...)
  
  // 4. 构造请求参数
  const params = {
    model: normalizeModelStringForAPI(options.model),
    messages: addCacheBreakpoints(messagesForAPI, ...),
    system,
    tools: allTools,
    max_tokens: maxOutputTokens,
    thinking: { type: 'adaptive' } | { type: 'enabled', budget_tokens: N },
    stream: true,
    ...extraBodyParams,
  }
  
  // 5. 通过 withRetry 包装发送请求
  const generator = withRetry(getClient, operation, retryOptions)
  const stream = await generator  // 获得 SSE 流
  
  // 6. 流式解析
  for await (const part of stream) {
    switch (part.type) {
      case 'message_start':
        // 记录 partialMessage, 计算 TTFT
        break
        
      case 'content_block_start':
        // 初始化 content block（text/tool_use/thinking）
        // tool_use: { type: 'tool_use', id, name, input: '' }
        // text:     { type: 'text', text: '' }
        // thinking: { type: 'thinking', thinking: '', signature: '' }
        break
        
      case 'content_block_delta':
        // 累积增量内容
        // input_json_delta → contentBlock.input += delta.partial_json
        // text_delta       → contentBlock.text += delta.text
        // thinking_delta   → contentBlock.thinking += delta.thinking
        // signature_delta  → contentBlock.signature = delta.signature
        break
        
      case 'content_block_stop':
        // 创建 AssistantMessage 并 yield
        const msg: AssistantMessage = {
          message: { ...partialMessage, content: normalizeContentFromAPI([contentBlock], tools) },
          type: 'assistant',
          uuid: randomUUID(),
          timestamp: new Date().toISOString(),
          requestId: streamRequestId,
        }
        yield msg
        break
        
      case 'message_delta':
        // 更新 usage, stop_reason, 计算 cost
        // 处理 max_tokens / model_context_window_exceeded
        break
        
      case 'message_stop':
        // no-op
        break
    }
    
    // 每个事件也 yield 为 StreamEvent（供 UI 实时显示）
    yield { type: 'stream_event', event: part }
  }
}
```

### 7.2 关键设计：每个 content_block_stop 都 yield 一个 AssistantMessage

这意味着一次 API 响应可能产生多个 `AssistantMessage`（例如一个 thinking block + 一个 text block + 多个 tool_use block，每个都是独立的 AssistantMessage）。

### 7.3 流式空闲看门狗

```typescript
const STREAM_IDLE_TIMEOUT_MS = 90_000  // 90秒无数据则中断
// 每收到一个 chunk 重置定时器
// 超时后释放流资源，触发 fallback 到非流式重试
```

---

## 8. 重试逻辑 (`withRetry`)

```typescript
const DEFAULT_MAX_RETRIES = 10
const BASE_DELAY_MS = 500
const MAX_529_RETRIES = 3

async function* withRetry(getClient, operation, options) {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const client = await getClient()
      const stream = await operation(client, attempt, retryContext)
      yield stream  // 成功则返回
      return
    } catch (error) {
      // 401/403: 刷新 OAuth token，重建 client
      if (error.status === 401 || error.status === 403) {
        await handleOAuth401Error()
        clearApiKeyHelperCache()
        continue
      }
      
      // 429: 速率限制
      if (error.status === 429) {
        const retryAfter = parseRetryAfterHeader(error)
        await sleep(retryAfter || exponentialBackoff(attempt))
        // 如果开启了 fast mode，降级到普通模式
        if (fastMode) { triggerFastModeCooldown() }
        continue
      }
      
      // 529: 过载
      if (error.status === 529) {
        if (retryCount529 >= MAX_529_RETRIES) {
          throw new FallbackTriggeredError(originalModel, fallbackModel)
        }
        await sleep(exponentialBackoff(attempt))
        retryCount529++
        continue
      }
      
      // ECONNRESET/EPIPE: 禁用 keep-alive 重连
      if (isStaleConnectionError(error)) {
        disableKeepAlive()
        continue
      }
      
      // max_tokens 溢出: 降低 maxTokensOverride
      if (isMaxTokensOverflow(error)) {
        retryContext.maxTokensOverride = adjustedValue
        continue
      }
      
      throw error  // 不可重试的错误
    }
    
    // yield 中间状态消息给 UI
    yield createSystemAPIErrorMessage(`Retrying (attempt ${attempt})...`)
    await sleep(exponentialBackoff(attempt))
  }
}

// 指数退避公式
function exponentialBackoff(attempt: number): number {
  return BASE_DELAY_MS * Math.pow(2, attempt) * (0.5 + Math.random() * 0.5)
}
```

### 持久重试模式（无人值守场景）

```typescript
// CLAUDE_CODE_UNATTENDED_RETRY 环境变量启用
// 对 429/529 无限重试，最大退避 5 分钟
// 每 30 秒发送心跳消息防止会话被判定为空闲
const PERSISTENT_MAX_BACKOFF_MS = 5 * 60 * 1000
const HEARTBEAT_INTERVAL_MS = 30_000
```

---

## 9. 工具执行详解

### 9.1 工具分区策略 (`toolOrchestration.ts`)

```typescript
function partitionToolCalls(toolUseMessages, toolUseContext): Batch[] {
  // 将工具调用分为多个批次：
  // - 连续的只读工具（isConcurrencySafe=true）合并为一个并发批次
  // - 每个写工具单独为一个串行批次
  // 示例：[Read, Read, Bash, Read, Edit] => [[Read,Read], [Bash], [Read], [Edit]]
}

async function* runTools(toolUseBlocks, assistantMessages, canUseTool, toolUseContext) {
  for (const { isConcurrencySafe, blocks } of partitionToolCalls(...)) {
    if (isConcurrencySafe) {
      // 并发执行（最多 10 个）
      yield* runToolsConcurrently(blocks, ...)
    } else {
      // 串行执行
      yield* runToolsSerially(blocks, ...)
    }
  }
}
```

### 9.2 流式工具执行 (`StreamingToolExecutor`)

核心思想：**模型还在流式输出时，已完成的 tool_use 块就开始执行**。

```typescript
class StreamingToolExecutor {
  private tools: TrackedTool[] = []
  private siblingAbortController: AbortController  // 子 abort，不影响主循环
  
  // 模型流式输出过程中，每完成一个 tool_use 块就调用
  addTool(block: ToolUseBlock, assistantMessage: AssistantMessage): void {
    const tracked = { id: block.id, block, status: 'queued', ... }
    this.tools.push(tracked)
    this.maybeStartExecution(tracked)
  }
  
  private maybeStartExecution(tool: TrackedTool): void {
    if (tool.isConcurrencySafe) {
      // 并发安全的工具：如果没有排他工具在执行，立即开始
      if (!this.hasExclusiveRunning()) {
        this.startTool(tool)
      }
    } else {
      // 排他工具：如果没有任何工具在执行，才开始
      if (!this.hasAnyRunning()) {
        this.startTool(tool)
      }
    }
  }
  
  // 获取已完成但未 yield 的结果（按添加顺序）
  getCompletedResults(): MessageUpdate[] { ... }
  
  // 等待所有剩余工具完成
  async* getRemainingResults(): AsyncGenerator<MessageUpdate> { ... }
  
  // Bash 出错时中止兄弟进程
  private onBashError(): void {
    this.siblingAbortController.abort()
    this.hasErrored = true
  }
}
```

### 9.3 单工具执行流程 (`toolExecution.ts`)

```typescript
async function* runToolUse(toolUse, assistantMessage, canUseTool, toolUseContext) {
  // 1. 查找工具定义
  const tool = findToolByName(tools, toolUse.name)
  
  // 2. 解析输入
  const parsedInput = tool.inputSchema.safeParse(toolUse.input)
  if (!parsedInput.success) {
    yield errorResult("Invalid input")
    return
  }
  
  // 3. 权限检查
  const permission = await canUseTool(tool, parsedInput.data, assistantMessage)
  if (permission.behavior === 'deny') {
    yield createUserMessage({ content: [{ type: 'tool_result', content: REJECT_MESSAGE, is_error: true, tool_use_id }] })
    return
  }
  
  // 4. 执行工具
  const result = await tool.call(parsedInput.data, toolUseContext, {
    abortSignal: toolUseContext.abortController.signal,
    onProgress: (progress) => { yield createProgressMessage(progress) },
  })
  
  // 5. 返回结果
  yield createUserMessage({
    content: [{ type: 'tool_result', content: result.output, is_error: result.isError, tool_use_id }],
    toolUseResult: result.output,
    sourceToolAssistantUUID: assistantMessage.uuid,
  })
  
  // 6. 更新上下文（如果工具修改了 context）
  if (result.contextModifier) {
    yield { newContext: result.contextModifier(toolUseContext) }
  }
}
```

---

## 10. 上下文管理 / Auto-Compact

### 10.1 阈值计算

```typescript
// autoCompact.ts
const MAX_OUTPUT_TOKENS_FOR_SUMMARY = 20_000

function getEffectiveContextWindowSize(model: string): number {
  return contextWindowForModel - min(modelMaxOutput, MAX_OUTPUT_TOKENS_FOR_SUMMARY)
}

const AUTOCOMPACT_BUFFER_TOKENS = 13_000
const MANUAL_COMPACT_BUFFER_TOKENS = 3_000

function getAutoCompactThreshold(model: string): number {
  return getEffectiveContextWindowSize(model) - AUTOCOMPACT_BUFFER_TOKENS
}

// 阻塞限制（手动 /compact 的保留空间）
blockingLimit = effectiveContextWindow - MANUAL_COMPACT_BUFFER_TOKENS
```

### 10.2 Auto-Compact 触发流程

```
每轮循环开始时：
1. applyToolResultBudget  → 裁剪过大的工具结果
2. snipCompactIfNeeded    → 轻量级历史裁剪
3. microcompactMessages   → 按 tool_use_id 移除旧 tool_result
4. applyCollapsesIfNeeded → 上下文折叠投影
5. autoCompactIfNeeded    → 如果 token 数 > threshold，触发完整压缩
```

### 10.3 压缩执行

```typescript
async function autoCompactIfNeeded(messages, toolUseContext, cacheSafeParams, ...) {
  const tokenCount = tokenCountWithEstimation(messages)
  const threshold = getAutoCompactThreshold(model)
  
  if (tokenCount < threshold) return { compactionResult: null }
  if (consecutiveFailures >= MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES) return { compactionResult: null }
  
  // 通过 forked agent 执行压缩
  const result = await compactConversation(messages, cacheSafeParams)
  // result 包含：summaryMessages, attachments, hookResults, preCompactTokenCount, postCompactTokenCount
  return { compactionResult: result }
}
```

### 10.4 压缩后消息重建

```typescript
function buildPostCompactMessages(compactionResult): Message[] {
  return [
    ...compactionResult.summaryMessages,    // 压缩摘要
    ...compactionResult.attachments,         // 保留的附件
    ...compactionResult.hookResults,         // hook 结果
  ]
}
```

### 10.5 断路器

连续 3 次 autocompact 失败后停止重试，防止无限循环浪费 API 调用。

---

## 11. Stop Hooks 机制

```typescript
async function* handleStopHooks(messages, assistantMessages, ...): StopHookResult {
  // 1. 执行注册的 stop hooks
  const hookResults = await executeStopHooks(context)
  
  // 2. 如果有阻塞错误（如 lint 失败、测试失败）
  if (hookResults.blockingErrors.length > 0) {
    return { blockingErrors: hookResults.blockingErrors, preventContinuation: false }
    // → 循环会将这些错误注入消息，让模型处理
  }
  
  // 3. 如果 hook 明确阻止继续
  if (hookResults.preventContinuation) {
    return { blockingErrors: [], preventContinuation: true }
  }
  
  // 4. 执行任务完成 hooks（提取记忆、提示建议等）
  await executeTaskCompletedHooks(...)
  
  return { blockingErrors: [], preventContinuation: false }
}
```

---

## 12. Generator 模式说明

整个 `query()` / `queryLoop()` 是一个 **AsyncGenerator**，通过 `yield` 向调用方（UI/SDK）推送事件：

```typescript
type YieldedTypes = 
  | StreamEvent            // 原始 SSE 事件（供 UI 实时渲染）
  | RequestStartEvent      // 请求开始标记
  | Message                // AssistantMessage / UserMessage / AttachmentMessage
  | TombstoneMessage       // 标记需要从 UI 移除的消息
  | ToolUseSummaryMessage  // 工具调用摘要（Haiku 生成）

type ReturnType = Terminal  // { reason: string; ... }
```

调用方消费方式：

```typescript
for await (const event of query(params)) {
  // 分发到 UI / 写入 transcript / 发送到 SDK
  dispatch(event)
}
// generator return 时获得 Terminal（退出原因）
```

---

## 13. 模型降级 (Fallback) 机制

```
1. 529 过载达到 MAX_529_RETRIES(3) 次
   → withRetry 抛出 FallbackTriggeredError
   
2. queryLoop 的 inner catch 捕获
   → 清空当前轮已产生的 assistantMessages/toolResults
   → yield tombstone 消息（移除 UI 中的不完整消息）
   → 切换 currentModel = fallbackModel
   → 设 attemptWithFallback = true, continue inner while

3. 用降级模型重试完整请求
```

---

## 14. 中断处理

```
用户按 Ctrl+C → toolUseContext.abortController.abort()

两个检查点：
1. 模型流式输出后 (STEP 5)
   → yield 中断消息
   → 如果有 StreamingToolExecutor，消费剩余结果（生成合成 tool_result）
   → return { reason: 'aborted_streaming' }

2. 工具执行后 (STEP 8)
   → yield 中断消息
   → return { reason: 'aborted_tools' }
   
特殊：submit-interrupt（用户提交了新消息）
   → 跳过中断消息（新消息自带上下文）
   → signal.reason === 'interrupt'
```

---

## 15. 关键设计决策总结

1. **AsyncGenerator 模式**：用 yield 推送所有事件，调用方自行消费。return 携带退出原因。
2. **状态不可变更新**：每次 `continue` 都创建新 `State` 对象赋值（非 mutate）。
3. **流式工具执行**：tool_use 块流完就开始执行，不等整个响应结束。
4. **多级上下文压缩**：snip → microcompact → collapse → autocompact，从轻到重。
5. **错误恢复梯度**：prompt-too-long 有 3 级恢复（collapse drain → reactive compact → 报错）；max_output_tokens 有 2 级恢复（escalate → resume injection × 3）。
6. **依赖注入**：`QueryDeps` 使核心循环可测试，测试可注入 fake callModel。
7. **断路器**：autocompact 连续失败 3 次后停止，防止死循环。
8. **幂等消息处理**：tombstone 机制确保降级/重试时 UI 不残留不完整消息。

---

## 16. 复刻指南：最小可行实现

要复刻此 Agent Loop 的核心逻辑，最小需要：

```typescript
// 1. 定义消息类型
interface Message { role: 'user' | 'assistant'; content: ContentBlock[] }

// 2. 定义核心循环
async function* agentLoop(params: {
  messages: Message[],
  systemPrompt: string,
  tools: Tool[],
  callModel: (messages, tools) => AsyncGenerator<AssistantMessage>,
  maxTurns?: number,
}) {
  let messages = [...params.messages]
  let turnCount = 0

  while (true) {
    turnCount++
    
    // 检查上下文大小，必要时压缩
    if (estimateTokens(messages) > THRESHOLD) {
      messages = await compact(messages)
    }
    
    // 调用模型
    const assistantMessages = []
    const toolUseBlocks = []
    for await (const msg of params.callModel(messages, params.tools)) {
      yield msg  // 推送给调用方
      assistantMessages.push(msg)
      toolUseBlocks.push(...extractToolUseBlocks(msg))
    }
    
    // 无工具调用 → 结束
    if (toolUseBlocks.length === 0) {
      return { reason: 'completed' }
    }
    
    // 执行工具
    const toolResults = []
    for (const toolUse of toolUseBlocks) {
      const result = await executeTool(toolUse, params.tools)
      yield result
      toolResults.push(result)
    }
    
    // 检查最大轮次
    if (params.maxTurns && turnCount >= params.maxTurns) {
      return { reason: 'max_turns' }
    }
    
    // 组装新消息继续循环
    messages = [...messages, ...assistantMessages, ...toolResults]
  }
}
```

然后逐步添加：
1. 错误恢复（max_output_tokens / prompt_too_long）
2. 重试逻辑（429/529/连接错误）
3. 流式工具执行
4. 多级上下文压缩
5. Stop hooks
6. 中断处理
7. 模型降级
