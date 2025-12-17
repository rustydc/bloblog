"""Example demonstrating Python logging integration with tinman.

This shows how to capture Python log messages as a tinman channel that can be
recorded and played back alongside other data.

Usage:
    # Run with automatic log capture
    tinman run examples.log_capture:pipeline --log-dir logs/ --capture-logs

    # Or run manually in Python
    python -m examples.log_capture
"""

import asyncio
import logging
from typing import Annotated

from tinman import In, Out, run
from tinman.logging import LogEntry, LogEntryCodec, LogHandler

# Set up application logger
logger = logging.getLogger("myapp")
logger.setLevel(logging.DEBUG)


async def data_producer(output: Annotated[Out[int], "data"]):
    """Producer that logs while producing data."""
    logger.info("Starting data production")
    
    for i in range(5):
        logger.debug(f"Producing value {i}")
        await output.publish(i)
        await asyncio.sleep(0.1)
    
    logger.info("Data production complete")


async def data_processor(
    input: Annotated[In[int], "data"],
    output: Annotated[Out[int], "processed"],
):
    """Processor that logs warnings for odd values."""
    logger.info("Processor starting")
    
    async for value in input:
        if value % 2 == 1:
            logger.warning(f"Odd value detected: {value}")
        result = value * 2
        logger.debug(f"Processed {value} -> {result}")
        await output.publish(result)
    
    logger.info("Processor complete")


# Global handler reference for closing
_handler: LogHandler | None = None


async def log_printer(logs: Annotated[In[LogEntry], "logs"]):
    """Consumer that prints captured log entries."""
    print("\n=== Captured Logs ===")
    async for entry in logs:
        level_name = logging.getLevelName(entry.level)
        print(f"[{level_name:8}] {entry.name}: {entry.message}")


async def results_consumer(input: Annotated[In[int], "processed"]):
    """Consume processed results and close handler when done."""
    results = []
    async for value in input:
        results.append(value)
    print(f"\nProcessed results: {results}")
    
    # Close the handler to signal log capture node to stop
    if _handler:
        _handler.close()


async def main():
    """Run the example with log capture."""
    global _handler
    
    # Create log handler and attach to our logger
    _handler = LogHandler(channel="app_logs", level=logging.DEBUG)
    logger.addHandler(_handler)
    
    # Also add a console handler for comparison
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(console)
    
    try:
        print("=== Console Output ===")
        await run([
            data_producer,
            data_processor,
            results_consumer,
            _handler.node,
            log_printer,
        ])
    finally:
        _handler.close()
        logger.removeHandler(_handler)
        logger.removeHandler(console)


# Pipeline for CLI usage
def create_pipeline():
    """Factory that creates the pipeline with log capture."""
    handler = LogHandler(channel="app_logs", level=logging.DEBUG)
    logging.getLogger("myapp").addHandler(handler)
    
    # Return nodes including the log capture node
    return [data_producer, data_processor, handler.node]


pipeline = [data_producer, data_processor]


if __name__ == "__main__":
    asyncio.run(main())
