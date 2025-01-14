import asyncio
import json
import re
from functools import cached_property
from pathlib import Path
from typing import Awaitable, Literal, overload

import yaml
from jsonpath_ng import parse

from vers.errors import VersError
from vers.task import Task
from vers.types.flows import (
    FlowInputSpec,
    FlowsSpec,
    FlowTaskSpec,
    JsonPath,
    SourceType,
)
from vers.types.media import MediaMessage
from vers.types.tasks import TaskOutput, TaskSpec, TasksSpec, UnstructuredTaskOutput


class Flow:
    LOADED_FLOWS_SPECS: dict[Path, FlowsSpec] = {}
    LOADED_TASKS_SPECS: dict[Path, TasksSpec] = {}

    def __init__(
        self,
        *,
        name: str,
        input_specs: list[FlowInputSpec],
        flow_task_specs: list[FlowTaskSpec],
        task_specs: list[TaskSpec],
        response_models: list[type[TaskOutput]],
    ) -> None:
        self.name = name
        self.input_specs = input_specs
        self.flow_task_specs = flow_task_specs
        self.task_specs = task_specs
        self.response_models_by_name = {
            response_model.__name__: response_model
            for response_model in response_models
        }
        self._tasks: dict[str, Task[TaskOutput]] | None = None

    @classmethod
    def load(
        cls,
        *,
        name: str,
        response_models: list[type[TaskOutput]],
        flows_spec_path: Path,
        tasks_spec_path: Path,
    ) -> "Flow":
        if flows_spec_path not in cls.LOADED_FLOWS_SPECS:
            with open(flows_spec_path) as file:
                yaml_content = yaml.safe_load(file)

            flows_spec = FlowsSpec.model_validate(yaml_content)
            cls.LOADED_FLOWS_SPECS[flows_spec_path] = flows_spec

        if tasks_spec_path not in cls.LOADED_TASKS_SPECS:
            with open(tasks_spec_path) as file:
                yaml_content = yaml.safe_load(file)

            tasks_spec = TasksSpec.model_validate(yaml_content)
            cls.LOADED_TASKS_SPECS[tasks_spec_path] = tasks_spec

        flow_specs = cls.LOADED_FLOWS_SPECS[flows_spec_path].flows
        flow_spec = next(
            (flow_spec for flow_spec in flow_specs if flow_spec.name == name), None
        )
        if flow_spec is None:
            raise VersError(
                error_type="VALIDATION_ERROR",
                message=f"Flow {name} not found in {flows_spec_path}",
            )

        task_specs_by_name = {
            task_spec.name: task_spec
            for task_spec in cls.LOADED_TASKS_SPECS[tasks_spec_path].tasks
        }
        task_specs_in_flow: list[TaskSpec] = []
        for flow_task_spec in flow_spec.tasks:
            task_spec = task_specs_by_name.get(flow_task_spec.name)
            if task_spec is None:
                raise VersError(
                    error_type="VALIDATION_ERROR",
                    message=f"Task {flow_task_spec.name} not found in {tasks_spec_path}",
                )

            task_specs_in_flow.append(task_spec)

        return cls(
            name=name,
            input_specs=flow_spec.inputs,
            flow_task_specs=flow_spec.tasks,
            task_specs=task_specs_in_flow,
            response_models=response_models,
        )

    async def run(
        self,
        *,
        run_id: str,
        disable_tracing: bool = False,
        **arguments: str | MediaMessage,
    ) -> None:
        for task_execution_level in self.task_execution_order:
            task_executions: list[Awaitable[None]] = []
            for task_name in task_execution_level:
                task = self.tasks[task_name]
                flow_task_spec = next(
                    (
                        flow_task_spec
                        for flow_task_spec in self.flow_task_specs
                        if flow_task_spec.name == task_name
                    ),
                    None,
                )
                if flow_task_spec is None:
                    raise VersError(
                        error_type="VALIDATION_ERROR",
                        message=f"Task {task_name} is not a valid task for this flow",
                    )

                task_executions.append(
                    self.__execute_task(
                        task=task,
                        flow_task_spec=flow_task_spec,
                        flow_argument_values=arguments,
                        run_id=run_id,
                        disable_tracing=disable_tracing,
                    )
                )

            await asyncio.gather(*task_executions)

    @overload
    def get[ResponseModelT: TaskOutput](
        self,
        task_name: str,
        /,
        *,
        response_model: type[ResponseModelT] = UnstructuredTaskOutput,
        key: str | None = None,
        optional: Literal[True],
    ) -> ResponseModelT | None: ...
    @overload
    def get[ResponseModelT: TaskOutput](
        self,
        task_name: str,
        /,
        *,
        response_model: type[ResponseModelT] = UnstructuredTaskOutput,
        key: str | None = None,
        optional: Literal[False],
    ) -> ResponseModelT: ...
    @overload
    def get[ResponseModelT: TaskOutput](
        self,
        task_name: str,
        /,
        *,
        response_model: type[ResponseModelT] = UnstructuredTaskOutput,
        key: str | None = None,
    ) -> ResponseModelT: ...

    def get[ResponseModelT: TaskOutput](
        self,
        task_name: str,
        /,
        *,
        response_model: type[ResponseModelT] = UnstructuredTaskOutput,
        key: str | None = None,
        optional: bool = False,
    ) -> ResponseModelT | None:
        task = self.tasks.get(task_name)

        if task is None:
            raise VersError(
                error_type="VALIDATION_ERROR",
                message=f"Task {task_name} is not a valid task for this flow",
            )

        if task.skipped:
            if optional:
                return
            else:
                raise VersError(
                    error_type="VALIDATION_ERROR",
                    message=f"Task {task_name} was skipped and not optional",
                )

        if key is None and len(task.outputs) > 1:
            raise VersError(
                error_type="VALIDATION_ERROR",
                message=f"Task {task_name} has multiple outputs but no key was provided",
            )

        keyed_output = task.outputs.get(key)
        if keyed_output is None:
            raise VersError(
                error_type="VALIDATION_ERROR",
                message=f"Task {task_name} does not have an output with key {key}",
            )

        if not isinstance(keyed_output, response_model):
            raise VersError(
                error_type="VALIDATION_ERROR",
                message=f"Task {task_name} output with key {key} is not of type {response_model}",
            )

        return keyed_output

    @property
    def tasks(self) -> dict[str, Task[TaskOutput]]:
        if self._tasks is None:
            tasks: dict[str, Task[TaskOutput]] = {}
            for task_spec in self.task_specs:
                if task_spec.response_model_name is not None:
                    response_model = self.response_models_by_name.get(
                        task_spec.response_model_name
                    )
                    if response_model is None:
                        raise VersError(
                            error_type="VALIDATION_ERROR",
                            message=f"Response model {task_spec.response_model_name} not found",
                        )

                    tasks[task_spec.name] = Task(
                        name=task_spec.name,
                        model=task_spec.model,
                        messages=task_spec.messages,
                        parameters=task_spec.parameters or [],
                        response_model=response_model,
                    )
                else:
                    tasks[task_spec.name] = Task(
                        name=task_spec.name,
                        model=task_spec.model,
                        messages=task_spec.messages,
                        parameters=task_spec.parameters or [],
                    )

            self._tasks = tasks

        return self._tasks

    @cached_property
    def task_execution_order(self) -> list[list[str]]:
        task_levels: dict[str, int] = {}
        visited: set[str] = set()
        tasks_by_name = {task.name: task for task in self.flow_task_specs}

        def calculate_depth(*, flow_task_spec: FlowTaskSpec) -> int:
            if flow_task_spec.name in visited:
                return task_levels[flow_task_spec.name]

            visited.add(flow_task_spec.name)
            if not flow_task_spec.inbound_task_names:
                depth = 0
            else:
                depth = (
                    max(
                        calculate_depth(flow_task_spec=tasks_by_name[inbound_task_name])
                        for inbound_task_name in flow_task_spec.inbound_task_names
                    )
                    + 1
                )

            task_levels[flow_task_spec.name] = depth
            return depth

        for flow_task_spec in self.flow_task_specs:
            calculate_depth(flow_task_spec=flow_task_spec)

        max_depth = max(task_levels.values())
        task_execution_order = [[] for _ in range(max_depth + 1)]
        for task_name in tasks_by_name:
            depth = task_levels[task_name]
            task_execution_order[depth].append(task_name)

        return task_execution_order

    async def __execute_task[ResponseModelT: TaskOutput](
        self,
        *,
        task: Task[ResponseModelT],
        flow_task_spec: FlowTaskSpec,
        flow_argument_values: dict[str, str | MediaMessage],
        run_id: str,
        disable_tracing: bool,
    ) -> None:
        task_arguments: dict[str, str | MediaMessage | list[str]] = {}
        for flow_argument_spec in flow_task_spec.arguments:
            task_arguments[flow_argument_spec.parameter] = (
                self.__get_task_argument_value(
                    parameter=flow_argument_spec.parameter,
                    source_type=flow_argument_spec.source_type,
                    source=flow_argument_spec.source,
                    path=flow_argument_spec.path,
                    flow_argument_values=flow_argument_values,
                    mapped=flow_argument_spec.mapped,
                )
            )

        for condition in flow_task_spec.conditions or []:
            match_value = (
                parse(condition.match_path)
                .find(task_arguments[condition.parameter])[0]
                .value
            )
            if not re.match(condition.match, str(match_value)):
                task.skipped = True
                return

        argument_sets: dict[str | None, dict[str, str | MediaMessage]] = {}
        mapped_arguments = [
            flow_argument_spec
            for flow_argument_spec in flow_task_spec.arguments
            if flow_argument_spec.mapped
        ]
        for mapped_argument in mapped_arguments:
            task_argument = task_arguments[mapped_argument.parameter]
            if not isinstance(task_argument, list):
                raise VersError(
                    error_type="VALIDATION_ERROR",
                    message="Cannot map non-iterable argument",
                )

            for value in task_argument:
                argument_set: dict[str, str | MediaMessage] = {}
                for parameter, current_argument in task_arguments.items():
                    if parameter == mapped_argument.parameter:
                        argument_set[parameter] = value
                    elif isinstance(current_argument, list):
                        raise VersError(
                            error_type="VALIDATION_ERROR",
                            message="Only one mapped argument is supported",
                        )
                    else:
                        argument_set[parameter] = current_argument

                argument_sets[value] = argument_set
        if len(argument_sets) == 0:
            argument_sets[None] = {}
            for parameter, argument in task_arguments.items():
                if isinstance(argument, list):
                    argument_sets[None][parameter] = json.dumps(argument)
                else:
                    argument_sets[None][parameter] = argument

        tasks_to_run: list[Awaitable[None]] = []
        for keyed_argument, argument_set in argument_sets.items():
            tasks_to_run.append(
                task.run(
                    flow_name=self.name,
                    run_id=run_id,
                    map_index=keyed_argument,
                    disable_tracing=disable_tracing,
                    **argument_set,
                )
            )

        await asyncio.gather(*tasks_to_run)

    def __get_task_argument_value(
        self,
        *,
        parameter: str,
        source_type: SourceType,
        source: str,
        path: JsonPath | None,
        flow_argument_values: dict[str, str | MediaMessage],
        mapped: bool,
    ) -> str | MediaMessage | list[str]:
        if source_type == "input":
            value = flow_argument_values.get(source)
            if value is None:
                raise VersError(
                    error_type="VALIDATION_ERROR",
                    message=f"Task requires argument {parameter} but was not provided to Flow",
                )
        else:
            task = self.tasks.get(source)
            if task is None:
                raise VersError(
                    error_type="VALIDATION_ERROR",
                    message=f"Task has source {source} but that task does not exist in this flow",
                )
            if len(task.outputs) > 1:
                if path is not None:
                    parse_result = parse(path).find(
                        {
                            key: output.model_dump()
                            for key, output in task.outputs.items()
                        }
                    )
                    parsed = [result.value for result in parse_result]
                else:
                    parsed = [output.model_dump() for output in task.outputs.values()]
            else:
                if path is not None:
                    parse_result = parse(path).find(task.output.model_dump())
                    parsed = [result.value for result in parse_result]
                else:
                    parsed = [task.output.model_dump()]

            if len(parsed) == 0:
                raise VersError(
                    error_type="VALIDATION_ERROR",
                    message=f"Argument {parameter} could not be parsed from task result using path {path}",
                )
            elif len(parsed) == 1:
                value = json.dumps(parsed[0])
            else:
                value = [json.dumps(result) for result in parsed]

            if mapped and isinstance(value, str):
                value = [subvalue for subvalue in json.loads(value)]
            elif isinstance(value, str):
                value = json.loads(value)

        return value
    

    async def cleanup(self) -> None:
        """Clean up any resources used by the Flow instance."""
        # Clear any cached tasks and their outputs
        if self._tasks is not None:
            # for task in self._tasks.values():
            #     task._outputs.clear()
            self._tasks.clear()
            self._tasks = None

        # Clear cached properties if any exist
        if hasattr(self, 'task_execution_order'):
            del self.task_execution_order

        # Clear the class-level caches if needed
        self.LOADED_FLOWS_SPECS.clear()
        self.LOADED_TASKS_SPECS.clear()

        # Explicitly close any event loops created by litellm
        # This helps prevent the "Event loop is closed" error
        import asyncio
        try:
            
            loop = asyncio.get_running_loop()
            if loop.is_closed():
                asyncio.set_event_loop(asyncio.new_event_loop())
        except RuntimeError:
            # If no loop is running, create a new one
            asyncio.set_event_loop(asyncio.new_event_loop())
