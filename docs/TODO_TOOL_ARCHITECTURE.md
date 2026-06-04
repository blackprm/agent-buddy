# Claude Code TodoWrite 工具完整实现文档

> 基于 claude-code 2.1.88 源码逆向分析，目标是让另一个 agent 可以完整复刻 TodoWrite 工具。

---

## 1. 文件清单

| 文件 | 职责 |
|------|------|
| `src/tools/TodoWriteTool/TodoWriteTool.ts` | 工具主体定义（call/schema/permissions） |
| `src/tools/TodoWriteTool/prompt.ts` | 工具 prompt 和 description |
| `src/tools/TodoWriteTool/constants.ts` | 工具名常量 `'TodoWrite'` |
| `src/utils/todo/types.ts` | TodoItem / TodoList 的 Zod schema |
| `src/state/AppStateStore.ts` | AppState 中 `todos: { [agentId]: TodoList }` |
| `src/utils/attachments.ts` | 提醒系统（`todo_reminder` 附件注入） |
| `src/utils/messages.ts` | 提醒消息格式化 |
| `src/utils/sessionRestore.ts` | 会话恢复时从 transcript 提取 todos |
| `src/components/Spinner.tsx` | UI：spinner 显示当前 `activeForm` |
| `src/components/TaskListV2.tsx` | UI：任务列表渲染 |

---

## 2. 数据模型

### 2.1 Zod Schema 定义

```typescript
// src/utils/todo/types.ts
import { z } from 'zod/v4'

const TodoStatusSchema = z.enum(['pending', 'in_progress', 'completed'])

export const TodoItemSchema = z.object({
  content: z.string().min(1, 'Content cannot be empty'),
  status: TodoStatusSchema,
  activeForm: z.string().min(1, 'Active form cannot be empty'),
})
export type TodoItem = z.infer<typeof TodoItemSchema>

export const TodoListSchema = z.array(TodoItemSchema)
export type TodoList = z.infer<typeof TodoListSchema>
```

### 2.2 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `content` | string | 祈使句形式，描述要做什么（如 "Run tests"） |
| `status` | enum | `pending` / `in_progress` / `completed` |
| `activeForm` | string | 现在进行时形式，执行时显示（如 "Running tests"） |

### 2.3 工具输入 Schema

```typescript
const inputSchema = z.strictObject({
  todos: TodoListSchema.describe('The updated todo list'),
})
```

**关键设计：每次调用传入完整的 todo 列表（全量替换），不是增量操作。**

---

## 3. 工具定义

### 3.1 工具名与配置

```typescript
export const TODO_WRITE_TOOL_NAME = 'TodoWrite'

export const TodoWriteTool = buildTool({
  name: TODO_WRITE_TOOL_NAME,
  searchHint: 'manage the session task checklist',
  maxResultSizeChars: 100_000,
  strict: true,              // 严格 schema 验证
  shouldDefer: true,         // 低优先级（在 tool search 中延迟加载）
  
  isEnabled() {
    return !isTodoV2Enabled()  // 交互模式下禁用（V2 Task 系统接管）
  },
  
  userFacingName() { return '' },          // 不在 UI 中显示工具名
  renderToolUseMessage() { return null },  // 不在聊天中渲染调用信息
  
  async checkPermissions(input) {
    return { behavior: 'allow', updatedInput: input }  // 始终允许，无需权限检查
  },
})
```

### 3.2 description（短描述，供模型理解工具用途）

```
Update the todo list for the current session. To be used proactively and often to track progress and pending tasks. Make sure that at least one task is in_progress at all times. Always provide both content (imperative) and activeForm (present continuous) for each task.
```

### 3.3 prompt（完整 181 行，核心规则提取）

**使用时机：**
1. 复杂多步任务（3 步以上）
2. 非平凡任务（需要规划）
3. 用户明确要求
4. 用户提供了多个任务
5. 收到新指令时立即捕获
6. 开始工作前标记 `in_progress`
7. 完成后立即标记 `completed`

**不使用时机：**
1. 单一简单任务
2. 少于 3 个简单步骤
3. 纯对话/信息性请求

**任务管理规则：**
- 始终保持恰好 1 个任务为 `in_progress`
- 完成后立即标记（不要批量更新）
- 未完成的任务不能标记为 `completed`
- 不再相关的任务直接从列表中移除
- 每个任务必须同时提供 `content`（祈使句）和 `activeForm`（进行时）

---

## 4. 执行逻辑 (`call` 函数)

```typescript
async call({ todos }, context) {
  const appState = context.getAppState()
  const todoKey = context.agentId ?? getSessionId()  // 按 agent/session 隔离
  const oldTodos = appState.todos[todoKey] ?? []
  
  // 全部完成时自动清空
  const allDone = todos.every(t => t.status === 'completed')
  const newTodos = allDone ? [] : todos

  // 验证提醒：主线程关闭 3+ 任务且无验证步骤时，提醒生成验证 agent
  let verificationNudgeNeeded = false
  if (
    !context.agentId &&              // 仅主线程
    allDone &&                        // 全部标记完成
    todos.length >= 3 &&              // 3+ 个任务
    !todos.some(t => /verif/i.test(t.content))  // 无验证步骤
  ) {
    verificationNudgeNeeded = true
  }

  // 更新 AppState
  context.setAppState(prev => ({
    ...prev,
    todos: {
      ...prev.todos,
      [todoKey]: newTodos,
    },
  }))

  return {
    data: { oldTodos, newTodos: todos, verificationNudgeNeeded },
  }
}
```

### 4.1 返回给模型的 tool_result

```typescript
mapToolResultToToolResultBlockParam({ verificationNudgeNeeded }, toolUseID) {
  const base = 'Todos have been modified successfully. Ensure that you continue to use the todo list to track your progress. Please proceed with the current tasks if applicable'
  const nudge = verificationNudgeNeeded
    ? '\n\nNOTE: You just closed out 3+ tasks and none of them was a verification step. Before writing your final summary, spawn the verification agent...'
    : ''
  return {
    tool_use_id: toolUseID,
    type: 'tool_result',
    content: base + nudge,
  }
}
```

---

## 5. 状态管理

### 5.1 AppState 存储

```typescript
// AppState 定义中
interface AppState {
  // ...其他字段
  todos: { [agentId: string]: TodoList }
}

// 初始化
const initialState: AppState = {
  todos: {},  // 空对象
}
```

### 5.2 存储键策略

```typescript
const todoKey = context.agentId ?? getSessionId()
```

- 主线程：用 `sessionId` 作为键
- 子 agent：用 `agentId` 作为键
- 每个 agent/session 有独立的 todo 列表

### 5.3 Agent 清理

当 agent 完成时，从 AppState 中移除其 todo：

```typescript
// runAgent.ts 中 agent 结束时
rootSetAppState(prev => {
  if (!(agentId in prev.todos)) return prev
  const { [agentId]: _removed, ...todos } = prev.todos
  return { ...prev, todos }
})
```

### 5.4 会话恢复

从 transcript 中提取最后一次 TodoWrite 调用的内容：

```typescript
function extractTodosFromTranscript(messages: Message[]): TodoList {
  // 从后往前扫描 assistant 消息
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]
    if (msg?.type !== 'assistant') continue
    
    // 找到 TodoWrite 的 tool_use block
    const toolUse = msg.message.content.find(
      block => block.type === 'tool_use' && block.name === 'TodoWrite'
    )
    if (!toolUse) continue
    
    // 解析 input.todos
    const parsed = TodoListSchema.safeParse(toolUse.input.todos)
    return parsed.success ? parsed.data : []
  }
  return []
}
```

---

## 6. 提醒系统

### 6.1 配置

```typescript
export const TODO_REMINDER_CONFIG = {
  TURNS_SINCE_WRITE: 10,       // 距上次 TodoWrite 超过 10 轮
  TURNS_BETWEEN_REMINDERS: 10, // 两次提醒间隔至少 10 轮
} as const
```

### 6.2 触发逻辑

```typescript
async function getTodoReminderAttachments(messages, toolUseContext): Promise<Attachment[]> {
  // 跳过条件
  if (TodoWrite 不在可用工具中) return []
  if (从未使用过 TodoWrite) return []
  if (当前 todo 为空) return []

  const { turnsSinceLastTodoWrite, turnsSinceLastReminder } = getTodoReminderTurnCounts(messages)

  if (
    turnsSinceLastTodoWrite >= 10 &&  // 已 10 轮没用 TodoWrite
    turnsSinceLastReminder >= 10      // 上次提醒已过 10 轮
  ) {
    const todos = appState.todos[todoKey] ?? []
    return [{
      type: 'todo_reminder',
      content: todos,
      itemCount: todos.length,
    }]
  }
  return []
}
```

### 6.3 提醒消息内容

```typescript
// 注入为 <system-reminder> 标签包裹的消息
case 'todo_reminder': {
  const todoItems = attachment.content
    .map((todo, i) => `${i + 1}. [${todo.status}] ${todo.content}`)
    .join('\n')

  let message = `The TodoWrite tool hasn't been used recently. If you're working on tasks that would benefit from tracking progress, consider using the TodoWrite tool to track progress. Also consider cleaning up the todo list if has become stale and no longer matches what you are working on. Only use it if it's relevant to the current work. This is just a gentle reminder - ignore if not applicable. Make sure that you NEVER mention this reminder to the user`
  
  if (todoItems.length > 0) {
    message += `\n\nHere are the existing contents of your todo list:\n\n[${todoItems}]`
  }
  
  return wrapMessagesInSystemReminder([createUserMessage({ content: message, isMeta: true })])
}
```

### 6.4 注入时机

提醒在 `getAttachmentMessages()` 中被收集，在 agent loop 的每轮工具执行完毕后注入（即 query.ts STEP 9）。

---

## 7. UI 渲染

### 7.1 Spinner 显示当前任务

```typescript
// Spinner.tsx
const currentTodo = tasksV2?.find(task => task.status !== 'pending' && task.status !== 'completed')
const nextTask = findNextPendingTask(tasksV2)

// 优先级：覆盖消息 > 当前任务的 activeForm > 当前任务的 subject > 随机动词
const leaderVerb = overrideMessage ?? currentTodo?.activeForm ?? currentTodo?.subject ?? randomVerb
const message = leaderVerb + '…'  // 例如 "Running tests…"

// 底部提示下一个任务
// "Next: Build the project"
```

### 7.2 任务列表渲染 (TaskListV2.tsx)

```
图标：
  ✓ completed (绿色)
  ■ in_progress (品牌色)
  □ pending (灰色)

截断策略：任务过多时，优先显示：
  1. 最近完成的
  2. 正在进行的
  3. 待处理的
```

---

## 8. V1 vs V2 系统切换

```typescript
export function isTodoV2Enabled(): boolean {
  if (isEnvTruthy(process.env.CLAUDE_CODE_ENABLE_TASKS)) return true
  return !getIsNonInteractiveSession()  // 交互模式用 V2
}
```

| 特性 | V1 (TodoWrite) | V2 (Task 工具) |
|------|----------------|----------------|
| 适用场景 | 非交互/SDK 会话 | 交互式会话 |
| 存储方式 | 内存（AppState） | 文件（~/.claude/tasks/） |
| 工具名 | `TodoWrite` | `TaskCreate/TaskUpdate/TaskList/TaskGet` |
| 数据模型 | content/status/activeForm | id/subject/description/owner/blocks/metadata |
| 并发支持 | 按 agentId 隔离 | 文件锁 + claiming |
| 持久化 | 仅会话内（恢复靠 transcript） | 磁盘持久化 |

---

## 9. 关键设计决策

1. **全量替换**：每次 TodoWrite 传入完整列表，不做增量 patch。简化实现但增加 token 消耗。
2. **双形态描述**：`content`（祈使句给人看） + `activeForm`（进行时给 spinner 显示）。
3. **自动清空**：全部 completed 时存储清空为 `[]`，避免内存泄漏。
4. **无权限检查**：TodoWrite 始终允许执行，因为它只修改内存状态不影响文件系统。
5. **不可见工具调用**：`userFacingName=''`, `renderToolUseMessage=null`，用户不会在聊天中看到 TodoWrite 的调用。
6. **隐式提醒**：超过 10 轮不用就注入提醒，但明确要求模型"不要向用户提及此提醒"。
7. **验证催促**：关闭 3+ 任务时如果没有验证步骤，催促模型生成 verification agent。
8. **会话恢复**：从 transcript 反向扫描最后一次 TodoWrite 调用来重建状态。

---

## 10. 复刻指南：最小可行实现

```typescript
// === 类型定义 ===
type TodoStatus = 'pending' | 'in_progress' | 'completed'
type TodoItem = { content: string; status: TodoStatus; activeForm: string }
type TodoList = TodoItem[]

// === 状态存储 ===
const todoStore: Map<string, TodoList> = new Map()

// === 工具定义 ===
const TodoWriteTool = {
  name: 'TodoWrite',
  description: 'Update the todo list for the current session...',
  inputSchema: {
    type: 'object',
    properties: {
      todos: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            content: { type: 'string' },
            status: { type: 'string', enum: ['pending', 'in_progress', 'completed'] },
            activeForm: { type: 'string' },
          },
          required: ['content', 'status', 'activeForm'],
        },
      },
    },
    required: ['todos'],
  },
  
  call(input: { todos: TodoList }, sessionId: string): string {
    const allDone = input.todos.every(t => t.status === 'completed')
    todoStore.set(sessionId, allDone ? [] : input.todos)
    return 'Todos have been modified successfully. Please proceed with the current tasks if applicable.'
  },
}

// === UI 集成 ===
function getSpinnerMessage(sessionId: string): string {
  const todos = todoStore.get(sessionId) ?? []
  const current = todos.find(t => t.status === 'in_progress')
  return current ? current.activeForm + '...' : 'Thinking...'
}

// === 提醒系统 ===
function shouldRemind(turnsSinceLastWrite: number, turnsSinceLastReminder: number): boolean {
  return turnsSinceLastWrite >= 10 && turnsSinceLastReminder >= 10
}

function getReminderMessage(todos: TodoList): string {
  const items = todos.map((t, i) => `${i + 1}. [${t.status}] ${t.content}`).join('\n')
  return `The TodoWrite tool hasn't been used recently. Consider using it to track progress.\n\n${items}`
}

// === 会话恢复 ===
function restoreTodos(transcript: Message[]): TodoList {
  // 从后往前找最后一次 TodoWrite 调用
  for (let i = transcript.length - 1; i >= 0; i--) {
    if (transcript[i].role === 'assistant') {
      const toolUse = findToolUseBlock(transcript[i], 'TodoWrite')
      if (toolUse) return toolUse.input.todos
    }
  }
  return []
}
```

### 集成到 Agent Loop 的要点

1. **注册为工具**：加入 tools 列表，schema 传给 API
2. **始终允许执行**：权限检查直接返回 allow
3. **结果不可见**：不在聊天 UI 中显示工具调用
4. **Spinner 集成**：读取当前 in_progress 任务的 activeForm 作为 spinner 文字
5. **提醒注入**：在 agent loop 每轮附件阶段检查是否需要注入提醒
6. **会话恢复**：resume 时从 transcript 提取最后状态

---

## 11. 完整 Prompt 文本（供复制）

```
Use this tool to create and manage a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.
It also helps the user understand the progress of the task and overall progress of their requests.

## When to Use This Tool
Use this tool proactively in these scenarios:

1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests todo list - When the user directly asks you to use the todo list
4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
5. After receiving new instructions - Immediately capture user requirements as todos
6. When you start working on a task - Mark it as in_progress BEFORE beginning work. Ideally you should only have one todo as in_progress at a time
7. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation

## When NOT to Use This Tool

Skip using this tool when:
1. There is only a single, straightforward task
2. The task is trivial and tracking it provides no organizational benefit
3. The task can be completed in less than 3 trivial steps
4. The task is purely conversational or informational

NOTE that you should not use this tool if there is only one trivial task to do. In this case you are better off just doing the task directly.

## Task States and Management

1. **Task States**: Use these states to track progress:
   - pending: Task not yet started
   - in_progress: Currently working on (limit to ONE task at a time)
   - completed: Task finished successfully

   **IMPORTANT**: Task descriptions must have two forms:
   - content: The imperative form describing what needs to be done (e.g., "Run tests", "Build the project")
   - activeForm: The present continuous form shown during execution (e.g., "Running tests", "Building the project")

2. **Task Management**:
   - Update task status in real-time as you work
   - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
   - Exactly ONE task must be in_progress at any time (not less, not more)
   - Complete current tasks before starting new ones
   - Remove tasks that are no longer relevant from the list entirely

3. **Task Completion Requirements**:
   - ONLY mark a task as completed when you have FULLY accomplished it
   - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
   - When blocked, create a new task describing what needs to be resolved
   - Never mark a task as completed if:
     - Tests are failing
     - Implementation is partial
     - You encountered unresolved errors
     - You couldn't find necessary files or dependencies

4. **Task Breakdown**:
   - Create specific, actionable items
   - Break complex tasks into smaller, manageable steps
   - Use clear, descriptive task names
   - Always provide both forms:
     - content: "Fix authentication bug"
     - activeForm: "Fixing authentication bug"

When in doubt, use this tool. Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully.
```
