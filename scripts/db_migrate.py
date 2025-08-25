"""
Database migration script.
This script uses Alembic to handle database migrations.
"""

import subprocess
import sys


def run_command(command: str) -> str:
    """Run a shell command and print its output."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result.stdout


def main():
    """Main function to handle database migrations."""
    import argparse

    parser = argparse.ArgumentParser(description="Database migration script.")
    parser.add_argument("action", choices=["init", "migrate", "upgrade", "downgrade", "current", "history"])
    parser.add_argument("--message", "-m", help="Migration message for 'migrate' action.")
    parser.add_argument("--revision", "-r", help="Revision identifier for 'downgrade' action.")
    args = parser.parse_args()

    if args.action == "init":
        print("Initializing Alembic environment...")
        run_command("alembic init alembic")

    elif args.action == "migrate":
        if not args.message:
            print("Error: Migration message is required for 'migrate' action.")
            sys.exit(1)
        print(f"Creating new migration with message: {args.message}")
        run_command(f'alembic revision --autogenerate -m "{args.message}"')

    elif args.action == "upgrade":
        revision = args.revision if args.revision else "head"
        print(f"Upgrading database to revision: {revision}")
        run_command(f"alembic upgrade {revision}")

    elif args.action == "downgrade":
        revision = args.revision if args.revision else "-1"
        print(f"Downgrading database to revision: {args.revision}")
        run_command(f"alembic downgrade {args.revision}")

    elif args.action == "current":
        print("Current database revision:")
        output = run_command("alembic current")
        print(output)

    elif args.action == "history":
        print("Database migration history:")
        output = run_command("alembic history")
        print(output)


if __name__ == "__main__":
    main()
