### Critical Rules - DO NOT VIOLATE
- **NEVER create mock data or simplified components** unless explicitly told to do so
- **NEVER replace existing complex components with simplified versions** - always fix the actual problem
- **ALWAYS work with the existing codebase** - do not create new simplified alternatives
- **ALWAYS find and fix the root cause** of issues instead of creating workarounds
- When debugging issues, focus on fixing the existing implementation, not replacing it
- When something doesn't work, debug and fix it - don't start over with a simple version
- If a specific tool or python package doesn't work, use the context7 MCP to search documentation.
- When writing Playwright tests, use the Playwright MCP to first visit the pages as an LLM and then remember and construct the Python test after successfully manually testing the page.