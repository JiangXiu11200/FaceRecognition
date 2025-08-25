# Database Migrations with Alembic

This directory contains database migration scripts for the Face Recognition System.

## Quick Start

### Apply migrations

```bash
alembic upgrade head 
```

Or you can migrate it though `uv` tool

```bash
uv run alembic upgrade head
```

> Hite: The following content will omit the `uv` tool execution command. If you want to use it, just add `uv run ...`.


### Rollback migrations

Rollback one version

```bash
alembic downgrade -1
```

Rollback all migrations

```bash
alembic downgrade base
```

### Common Commands

| Command | Description |
| ------- | ----------- |
| ```alembic current``` | Show current revision |
| ```alembic history``` | Show migration history |
| ```alembic upgrade head``` | Apply all migrations |
| ```alembic downgrade -1``` | Rollback last migration |
| ```alembic revision --autogenerate -m "message"``` | Auto-generate migration |
| ```alembic revision -m "message"``` | Create empty migration |


### Project Specific Notes

Database Configuration

- Development: SQLite (`face_recognition.db`)
- Production: PostgreSQL (configured via environment variables)

