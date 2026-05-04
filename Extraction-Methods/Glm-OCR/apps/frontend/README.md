# Z-OCR Frontend

OCR 前端应用，支持 PDF 和图片文件的智能识别与解析，提供流畅的用户体验和强大的结果展示功能。

## ✨ 功能特性

- 📄 **多格式支持**：支持 PDF、PNG、JPEG 等格式文件上传
- 🔍 **智能 OCR**：自动识别文档中的文字、表格、数学公式等内容
- 📊 **双栏布局**：左侧文件预览，右侧 OCR 结果展示
- 📝 **Markdown 渲染**：支持 Markdown 格式的结果展示，包含数学公式渲染（KaTeX）
- 🎯 **交互式预览**：PDF 预览与 OCR 结果块的双向定位和高亮
- ⚡ **性能优化**：虚拟滚动技术，支持大文档的高效渲染
- 📱 **响应式设计**：适配不同屏幕尺寸

## 🚀 快速开始

### 环境要求

- Node.js >= 18
- pnpm >= 8

### 安装依赖

```bash
pnpm install
```

### 开发环境

```bash
pnpm dev
```

应用将在 `http://localhost:3006` 启动。

### 构建生产版本

```bash
pnpm build
```

### 预览生产构建

```bash
pnpm preview
```

## 🧪 测试

项目使用 [Vitest](https://vitest.dev/) 进行测试：

```bash
pnpm test
```

## 🛠️ 技术栈

### 核心框架
- **React 19** - UI 框架
- **TypeScript** - 类型安全
- **Vite** - 构建工具

### 路由与状态管理
- **TanStack Router** - 文件路由系统
- **Zustand** - 轻量级状态管理

### UI 与样式
- **Tailwind CSS** - 实用优先的 CSS 框架
- **Shadcn UI** - 高质量组件库
- **Radix UI** - 无样式组件基元
- **Lucide React** - 图标库

### 功能库
- **React Markdown** - Markdown 渲染
- **KaTeX** - 数学公式渲染
- **React PDF** - PDF 预览
- **Axios** - HTTP 客户端
- **Sonner** - Toast 通知

## 📁 项目结构

```
src/
├── components/          # 组件目录
│   ├── ocr/            # OCR 相关组件
│   │   ├── MarkdownPreview.tsx    # Markdown 预览组件
│   │   ├── PdfViewer.tsx          # PDF 查看器
│   │   ├── CopyButton.tsx         # 复制按钮
│   │   └── JsonPreview.tsx        # JSON 预览
│   └── ui/             # UI 基础组件
├── routes/             # 路由页面
│   └── _ocr/          # OCR 相关路由
│       ├── OCRPage.tsx           # 主页面
│       ├── FileUpload.tsx        # 文件上传组件
│       ├── FilePreview.tsx       # 文件预览组件
│       └── OCRResults.tsx        # OCR 结果展示
├── hooks/              # 自定义 Hooks
│   ├── useMdVirtualRendering.ts  # Markdown 虚拟滚动
│   ├── usePdfNavigation.ts       # PDF 导航
│   └── usePdfZoom.ts             # PDF 缩放
├── lib/                # 工具库
│   ├── api.ts         # API 接口
│   └── utils.ts       # 工具函数
└── store/             # 状态管理
    └── useOcrStore.ts # OCR 状态存储
```

## 🔧 配置

### 环境变量

创建 `.env` 文件并配置：

```env
VITE_API_URL=http://localhost:8000/api/v1
```

### API 配置

默认 API 地址为 `http://localhost:8000/api/v1`，可通过环境变量 `VITE_API_URL` 修改。

## 📖 使用说明

### 文件上传

1. 点击上传区域或拖拽文件到上传区域
2. 支持的文件格式：PDF、PNG、JPEG
3. 文件大小限制：20MB
4. 上传后自动创建 OCR 任务并开始处理

### OCR 结果查看

- **Markdown 预览**：自动渲染识别结果，支持数学公式
- **PDF 预览**：支持缩放、翻页等操作
- **交互定位**：点击结果块可定位到 PDF 对应位置
- **复制功能**：支持复制识别结果文本

### 虚拟滚动

对于大型文档，系统会自动启用虚拟滚动，只渲染可见区域的内容，提升性能。

## 🎨 组件使用

### 添加 Shadcn 组件

```bash
pnpm dlx shadcn@latest add [component-name]
```

例如：

```bash
pnpm dlx shadcn@latest add button
```

## 🐳 Docker 部署

项目包含 Dockerfile 和 nginx 配置，支持容器化部署：

```bash
docker build -t z-ocr-frontend .
docker run -p 3006:80 z-ocr-frontend
```

## 📝 开发指南

### 添加新路由

在 `src/routes` 目录下创建新文件，TanStack Router 会自动识别并生成路由。

### 状态管理

使用 Zustand 进行状态管理，相关 store 定义在 `src/store` 目录下。

### 样式规范

- 使用 Tailwind CSS 进行样式编写
- 遵循项目现有的设计系统
- 支持深色模式，使用 CSS 变量定义颜色

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目为私有项目。

## 🔗 相关链接

- [TanStack Router 文档](https://tanstack.com/router)
- [React 文档](https://react.dev)
- [Tailwind CSS 文档](https://tailwindcss.com)
- [Shadcn UI 文档](https://ui.shadcn.com)
- [KaTeX 文档](https://katex.org)
