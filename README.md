# Agent Buddy / Claude Code Study

这是一个围绕 **Claude Code / Agent Runtime / 多通道智能体** 的学习与实验工作区。仓库的核心实现位于 `my-agent-core/`，同时通过 Git submodule 保留了若干参考项目，方便对照学习 Claude Code、OpenClaw 以及相关 Agent 架构设计。

## 项目内容

```text
.
├── my-agent-core/            # 自研 Agent Runtime 核心与 Web 调试服务
├── my-study/                 # 从零实现 Agent Loop / Tool Use / Permission 的学习代码
├── learn-claude-code/        # Claude Code 学习资料，Git submodule
├── openclaw/                 # OpenClaw 参考实现，Git submodule
├── claude-code-sourcemap/    # Claude Code sourcemap 还原参考，Git submodule
├── main.py                   # 根目录实验入口
└── hello                     # 简单占位/实验文件
```

## 核心模块：my-agent-core

`my-agent-core` 是本仓库主要开发对象，目标是构建一个后端友好的通用 Agent Core，支持：

- Agent Runtime 循环
- 多模型适配：Fake / OpenAI / Anthropic / 火山方舟 Ark / DeepSeek 等
- Tool Registry 与内置工具
- 权限策略与工具调用审批
- Plan Mode
- Session / Memory / Task 管理
- Sandbox 与 Worktree 隔离
- Hooks 系统
- Prompt YAML 模板管理与 Admin UI
- Terminal Web UI 调试界面
- Feishu / 飞书 API 集成
- Feishu WebSocket 入站消息桥：用户给机器人发消息，Agent 自动处理并回复

### 重要目录

```text
my-agent-core/
├── pyproject.toml
├── src/
│   ├── agent_core/           # Agent 核心能力
│   │   ├── core/             # AgentRuntime 与事件流
│   │   ├── model/            # 模型适配器
│   │   ├── tools/            # 工具系统
│   │   ├── permissions/      # 权限策略
│   │   ├── context/          # System Prompt / Context 构建
│   │   ├── session/          # 会话存储
│   │   ├── memory/           # Session Memory
│   │   ├── tasks/            # 任务系统
│   │   ├── sandbox/          # 沙箱管理
│   │   ├── hooks/            # Hook 引擎
│   │   ├── users/            # 用户与组织
│   │   └── integrations/     # 第三方集成，如飞书
│   └── agent_server/         # FastAPI 服务与 Web UI
│       ├── app.py            # Terminal API / WebSocket / Feishu endpoints
│       ├── admin.py          # Admin API
│       ├── prompt_store.py   # YAML PromptStore
│       ├── runtime_factory.py
│       ├── prompts/          # Prompt YAML 模板
│       └── static/           # terminal.html / admin.html
└── tests/                    # 单元测试
```

## 快速开始

### 1. 克隆仓库

如果需要同时拉取参考项目 submodule：

```bash
git clone --recurse-submodules git@github.com:blackprm/agent-buddy.git
cd agent-buddy
```

如果已经普通 clone：

```bash
git submodule update --init --recursive
```

### 2. 安装 my-agent-core

```bash
cd my-agent-core
python -m venv .venv
source .venv/bin/activate
pip install -e '.[web,dev]'
```

如需使用 SearXNG/Search 相关能力，可额外安装：

```bash
pip install -e '.[search]'
```

### 3. 配置环境变量

复制示例配置：

```bash
cp .env.example .env
```

根据需要填写模型 Key，例如：

```bash
AGENT_MODEL_PROVIDER=ark
ARK_API_KEY=your_ark_api_key_here
MODEL_ID=your_model_id
```

如果只是本地跑通流程，也可以使用默认 fake 模型。

### 4. 启动 Web 调试服务

```bash
uvicorn agent_server.app:app --reload
```

默认访问：

- Terminal UI: <http://127.0.0.1:8000/terminal>
- Admin UI: <http://127.0.0.1:8000/admin>

首次启动时，如果没有设置稳定 token，服务会在控制台打印临时访问 token。建议本地设置：

```bash
export AGENT_TERMINAL_TOKEN=your_terminal_token
export AGENT_ADMIN_TOKEN=your_admin_token
```

## 飞书消息桥

`my-agent-core` 已实现飞书 WebSocket 入站消息桥，能力类似 OpenClaw 的 Feishu channel：

1. 在飞书开放平台创建自建应用。
2. 获取 App ID 与 App Secret。
3. 在 Terminal UI 的「飞书」弹窗中粘贴 App 凭证。
4. 点击「测试连接」。
5. 点击「启动收消息」。
6. 在飞书里给机器人发消息，Agent 会收到消息并自动回复。

相关接口：

```text
GET  /terminal/api/integrations/feishu
PUT  /terminal/api/integrations/feishu/app-credentials
POST /terminal/api/integrations/feishu/test
POST /terminal/api/integrations/feishu/bridge/start
POST /terminal/api/integrations/feishu/bridge/stop
GET  /terminal/api/integrations/feishu/bridge/logs
```

诊断日志会记录：WebSocket 启动、消息入队、消息解析、Agent 执行、回复 API 调用结果等，便于定位飞书机器人不回复的问题。

## Prompt 管理后台

Prompt 内容已支持 YAML 模板化管理：

```text
my-agent-core/src/agent_server/prompts/default.yaml
my-agent-core/src/agent_server/prompts/minimal.yaml
```

Admin UI 支持：

- 查看 Prompt 模板
- 编辑 YAML
- 预览渲染后的 system prompt
- 管理运行时配置

访问：

```text
http://127.0.0.1:8000/admin
```

## 运行测试

在 `my-agent-core/` 目录下执行：

```bash
pytest
```

只运行飞书相关测试：

```bash
pytest tests/test_feishu_integration.py
```

## 学习代码

`my-study/` 是轻量学习实现，适合从零理解 Agent 基础机制：

```text
my-study/
├── s01_agent_loop/code.py    # Agent Loop
├── s02_tool_use/code.py      # Tool Use
└── s03_permission/code.py    # Permission / Approval
```

## Submodules

本仓库把以下参考项目作为 submodule 管理：

```text
claude-code-sourcemap -> https://github.com/ChinaSiro/claude-code-sourcemap.git
learn-claude-code     -> https://github.com/shareAI-lab/learn-claude-code.git
openclaw              -> https://github.com/openclaw/openclaw.git
```

更新 submodule：

```bash
git submodule update --remote --recursive
```

查看 submodule 状态：

```bash
git submodule status --recursive
```

## 安全说明

仓库通过 `.gitignore` 排除了常见本地敏感和生成文件：

- `.env`
- `.venv/`
- `__pycache__/`
- `*.db` / `*.sqlite*`
- `node_modules/`
- 本地 IDE 配置

请不要提交真实 API Key、App Secret、Access Token 或本地数据库。

## 当前状态

这个仓库目前更像是一个「Agent Runtime 学习 + 实验 + 原型实现」工作区，而不是单一发布包。推荐主要从 `my-agent-core/` 入手运行和开发，使用 submodule 中的项目作为参考资料。
