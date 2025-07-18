# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

AgentVault is a newly initialized repository. When setting up this project, consider the following:

## Project Setup Commands

Since this is an empty repository, the first task will be to initialize it with the appropriate technology stack. Common initialization commands include:

### For Node.js/TypeScript projects:
```bash
npm init -y
npm install typescript @types/node --save-dev
npx tsc --init
```

### For Python projects:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt  # Once requirements.txt is created
```

### For Rust projects:
```bash
cargo init
cargo build
cargo test
```

## Development Guidelines

1. **Project Structure**: When creating the initial structure, follow the conventions of the chosen technology stack.

2. **Testing**: Set up a testing framework appropriate to the language:
   - Node.js: Jest, Vitest, or Mocha
   - Python: pytest or unittest
   - Rust: Built-in cargo test

3. **Git Configuration**: Always create a `.gitignore` file appropriate for the technology stack before the first commit.

## Current Permissions

Claude has the following bash command permissions in this repository:
- `ls`: List directory contents
- `find`: Search for files and directories
- `tree`: Display directory structure

## Important Notes

- This repository is connected to GitHub at: https://github.com/DwirefS/AgentVault.git
- The repository name "AgentVault" suggests it may be related to agent systems, AI agents, or secure storage
- Always verify the intended purpose and technology stack with the user before initializing the project structure