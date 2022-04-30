from typing import Generic, TypeVar, Iterable, List, Optional
from enum import Enum
from dataclasses import dataclass
import uuid
import unittest

TaskDataType = TypeVar('TaskDataType')


class TaskState(Enum):
    """
    任务所处的状态
    """
    TODO = 0
    SUCCESS = 1
    FAILED = 2


class BaseTask(Generic[TaskDataType]):
    """
    任务基类, 定义公共接口
    """

    def __init__(self, data: TaskDataType):
        """
        初始化任务
        @param data: 任务所依赖的输入数据和将要输出的数据 (此处为了便利不区分输入输出数据, 实际中可以拆分为输入和输出
        """
        self.data = data
        self.state = TaskState.TODO
        self.error: Optional[Exception] = None

    def is_dependencies_fulfill(self):
        return all(map(lambda task: task.state == TaskState.SUCCESS, self.get_dependencies()))

    def get_dependencies(self) -> Iterable['BaseTask']:
        """
        定义任务的依赖, 如无依赖返回空列表
        @return:
        """
        raise NotImplementedError()

    def execute(self):
        try:
            self._execute()
            self.state = TaskState.SUCCESS
        except Exception as e:
            self.error = e
            self.state = TaskState.FAILED
            raise e

    def _execute(self):
        """
        编写任务的具体工作内容
        """
        raise NotImplementedError()


class TaskPipeline:
    """
    任务管道的执行
    """
    def __init__(self):
        self._tasks: List[BaseTask] = []

    def is_finish(self):
        return all(map(lambda task: task.state != TaskState.TODO, self._tasks))

    def add_task(self, task: BaseTask):
        self._tasks.append(task)

    def add_tasks(self, tasks: Iterable[BaseTask]):
        self._tasks.extend(tasks)

    def run(self):
        """
        自动根据依赖关系完成任务执行工作
        """
        while not self.is_finish():
            for task in self._tasks:
                if task.is_dependencies_fulfill() and task.state == TaskState.TODO:
                    task.execute()


@dataclass()
class PrepareTaskData:
    base_dir: str
    machines: Iterable[str]


class PrepareTask(BaseTask[PrepareTaskData]):
    """
    该任务负责测试开始前的准备工作, 如创建文件, 配置机器等
    """
    def get_dependencies(self):
        """ 该任务无外部依赖, 直接返回空依赖列表"""
        return []

    def _execute(self):
        print(f"初始化文件系统: {self.data.base_dir}")
        for machine in self.data.machines:
            print(f"对即将使用的机器进行配置: {machine}")
        return self.data


@dataclass()
class SubmitHpcTaskData:
    hpc_job_ids: Iterable[str]


class SubmitHpcTask(BaseTask[SubmitHpcTaskData]):
    """
    该任务负责向hpc集群提交任务
    """

    prepareTask: PrepareTask

    def get_dependencies(self):
        """
        该任务的前置任务是 PrepareTask
        @return:
        """
        return [self.prepareTask]

    def _execute(self):
        print("从前置任务中读取配置好的本地目录和远程机器信息")

        self.data.hpc_job_ids = []
        for machine in self.prepareTask.data.machines:
            job_id = uuid.uuid4()
            print(f"将任务提交到机器 {machine} 上执行, 得到 job id {job_id}")
            self.data.hpc_job_ids.append(job_id)


class ProcessOutput(BaseTask[None]):
    """
    从提交的hpc任务中获取输出并处理 (该任务无输入输出, 只依赖于上游任务的结果)
    """
    submitHpcTask: SubmitHpcTask

    def get_dependencies(self):
        return [self.submitHpcTask]

    def _execute(self):
        for job_id in self.submitHpcTask.data.hpc_job_ids:
            print(f"从job {job_id} 中获取数据并开始处理, 上传结果到数据库中...")


def create_hpc_task_pipeline(base_dir: str, machines: Iterable[str]):
    pipeline = TaskPipeline()

    # 实例化准备任务
    prepareTask = PrepareTask(PrepareTaskData(base_dir=base_dir, machines=machines))

    # 实例化任务提交
    subHpcTask = SubmitHpcTask(SubmitHpcTaskData(hpc_job_ids=[]))
    subHpcTask.prepareTask = prepareTask  # 建立依赖关系

    # 实例化输出分析任务
    outputProcess = ProcessOutput(None)
    outputProcess.submitHpcTask = subHpcTask # 建立依赖关系

    # 将所有任务加入pipeline中并执行
    ## 任务的添加顺序并无关系, 因为实际执行取决于依赖的拓扑排序
    pipeline.add_tasks([outputProcess, prepareTask, subHpcTask])
    return pipeline


class PipelineTest(unittest.TestCase):

    def test_pipeline(self):
        """
        该测试只用于演示目的, 没有断言
        @return:
        """
        pipeline = create_hpc_task_pipeline('/my/workspace', ['machine1', 'machine2'])
        pipeline.run()


if __name__ == '__main__':
    unittest.main()
