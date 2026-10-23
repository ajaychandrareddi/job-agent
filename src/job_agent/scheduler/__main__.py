"""Run the scheduler process."""

import asyncio

from job_agent.scheduler.bootstrap import main

if __name__ == "__main__":
    asyncio.run(main())
