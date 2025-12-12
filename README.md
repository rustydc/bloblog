# Tinman

**Write simple async coroutines. Get automatic pub/sub orchestration, logging, and replay.**

Tinman is an async-first framework that wires together your coroutines into data pipelines. You write simple `async` functions with annotated channel parameters, and Tinman handles the rest:

- **Automatic wiring** - Pub/sub connections inferred from type annotations
- **Concurrent orchestration** - All nodes run concurrently, data flows asynchronously  
- **Zero-config logging** - Every channel automatically logged to efficient binary format, with optional custom encodings
- **Instant replay** - Trivially re-run any subset of nodes against recorded data

Stop writing pub/sub boilerplate and event loops; just write your async logic, annotate your channels, and let Tinman orchestrate everything.

## Installation

```bash
pip install tinman
```

Requires Python 3.12+

## Quick Example: Robotic Vacuum

Here's a simple robotic vacuum showing the wiring of nodes for sensor fusion, decision making, and motor control:

```python
import asyncio
from typing import Annotated

from tinman import In, Out, enable_pickle_codec, playback, run

async def sensor_node(sensor_out: Annotated[Out[SensorData], "sensors"]):
    for _ in range(50):  # Run for 5 seconds at 10Hz
        data = read_sensors()
        await sensor_out.publish(data)
        await asyncio.sleep(0.1)

async def perception_node(
    sensor_in: Annotated[In[SensorData], "sensors"],
    world_out: Annotated[Out[WorldState], "world_state"]
):
    position = [0.0, 0.0]
    async for sensors in sensor_in:
        position[1] += 2.0  # Move forward 2cm per tick
        await world_out.publish(
            WorldState(
                obstacle_ahead=(sensors.lidar_distance < 30),
                low_battery=sensors.battery_level < 25,
                position=(position[0], position[1])
            )
        )

async def planner_node(
    world_in: Annotated[In[WorldState], "world_state"],
    command_out: Annotated[Out[Command], "commands"]
):
    planner = Planner()
    async for world in world_in:
        cmd = planner.get_command(world)
        if cmd:
            await command_out.publish(cmd)

async def motor_node(command_in: Annotated[In[Command], "commands"]):
    """Execute motor commands"""
    async for cmd in command_in:
        send_motor_commands(cmd)


async def main():
    # Simpler serialization for trusted log streams.
    enable_default_pickle_codec()
    
    # Run the full pipeline.
    await run(
        [sensor_node, perception_node, planner_node, motor_node],
        log_dir=Path("vacuum_logs")
    )
    
    # Run a different planner, and motor, using the above perception output.
    await playback(
        [aggressive_planner, motor_node],
        playback_dir=Path("vacuum_logs")
    )


if __name__ == "__main__":
    asyncio.run(main())
```

**What's happening:**
* **Sensors** → **Perception** → **Planner** → **Motors** pipeline
* First run: logs all channels (`sensors`, `world_state`, `commands`) 
* Second run: replays `world_state` from log, tests new planner with identical perception data
