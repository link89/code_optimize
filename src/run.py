import os
import struct
import numpy as np
import re

from functools import partial
from pathlib import Path
import copy
from types import FunctionType


def create_path(path_list, parents=True, exit_ok=False):

    """
    创建path_list列表里包含的绝对路径
    """

    for path in path_list:
        Path(path).mkdir(parents=parents, exist_ok=exit_ok) 


def dump_input(text, task_path, input_name, restart_dir=False):
    """
    输出text包含的文本到task_path路径下
    """

    fp = Path(task_path).absolute().joinpath(input_name)
    
    if not fp.exists():
        with open(fp, "w") as f:
            f.write(text)
        print(f"success: dumped {str(fp)}")
    else:
        raise FileExistsError(f"{fp} already exists!")

    if restart_dir:
        if restart_dir == True: #restart为将要创建在工作目录下的restart路径名
            restart_dir= "restart"
        Path(task_path).absolute().joinpath(restart_dir).mkdir(exist_ok=True)
        print(f"{restart_dir} path has been created.")



def dummy_submit(task_or_task_list, disp_resource):
    """
    上传计算任务的方法，这里省略上传过程
    """
    #mock function for uploading calculation task to HPC and retrieve the results.

    print("finished submission tasks")

    return True


def dummy_make_func(structure, cell, basis_set, calc_type):

    """
    用于生成cp2k输入文件的方法, 省略了cp2k输入文件的内容，包括可能的参数传入
    """

    #mock function to generate a input file for science calculations.
    txt = f"{structure}\n{cell}\n{basis_set}\n{calc_type}"

    return txt

class RunBase():


    """
    用于描述计算任务的基类， 包含了 制作输入文件、提交计算任务、后处理等步骤，有待优化
    """

    # an python object illustrating the science calculation:
    #   1. make the input files for calc, 
    #   2. submit the calc and retrieve the results. 
    #   3. some postprocess with the results. 


    def __init__(self,
        work_base: str,
        task_dir: str,
        make_func: FunctionType,

        previous_file: list=list(),
        disp_resource: dict=None,
        mk_restart: str or bool= "./restart",
        input_name: str = 'untitiled.input',
        post_func= None,
        forward_common_files: list=list(),
        backward_common_files:list=list(),
        run_name=None,
        **kwargs):

        pass


        self.work_base = Path(work_base).absolute()

        self.set_task_dir(task_dir)

        saved_previous_file_dirs = []
        for _pre_dir in previous_file: #BUG previous_file 必须是list
            print(_pre_dir)

            if os.path.isabs(_pre_dir):
                saved_previous_file_dirs.append(Path(_pre_dir))
            else:
                saved_previous_file_dirs.append(Path(self.work_base/(_pre_dir))) # BUG, 注意这里是相对于work_base的路径 来生成绝对路径
        self.previous_file = saved_previous_file_dirs

        self.run_name = run_name

        self.make_func = make_func
        self.input_name = input_name
        self.mk_restart = mk_restart
        self.prepare_tag = False #prepare中包含创建文件夹和输出input过程 是一次性的操作

        self.post_func = post_func

        self.forward_common_files = forward_common_files
        self.backward_common_files = backward_common_files

        self._load_resource(disp_resource)


    def set_task_dir(self, new_task_dir):

        """
        设置对应计算任务的路径， 包含任务的绝对路径和 相对于work_base的相对路径
        """
        #检查task_dir路径相对还是绝对，最后用绝对的形式保存

        if os.path.isabs(new_task_dir):
            self.task_dir = Path(new_task_dir)
        else:
            self.task_dir = Path(self.work_base/(new_task_dir))

        print(self.task_dir)
        self.rela_task_dir = self.task_dir.relative_to(self.work_base)


    def set_structure(self, new_structure, cell=None):

        """
        设置任务对象所需要的化学结构文件，晶胞信息可以不包括
        """

        self.struct = new_structure
        if cell:
            self.struct.set_cell(cell)



    def _render_make_func(self, *make_args, **make_kwargs):

        """
        用来配置 输出input函数的参数方法，对参数进行“预定义”
        """

        self.make_func = partial(self.make_func,*make_args, **make_kwargs)


    def _check_make_function(self):

        """
        检查产生输入文件的函数是否可以正常运行
        """

        try:
            self.make_func() 
        except:
            raise Exception("make function in RunMeta has not been prepared yet!, please check the make function.")
        pass


    def _load_resource(self, resource):

        """
        加载提交计算任务需要的配置信息
        """

        self.submit_resource = resource

    def _prepare_task(self, **task_kwargs):

        """
        准备计算任务信息
        """
        pass


    def _dump_struct(self):

        """
        在任务路径下输出化学计算需要的结构文件
        """

        raise NotImplementedError("method not implement")
    
    def _dump_input(self):

        """
        在任务路径下输出化学计算需要的输入文件
        """

        raise NotImplementedError("method not implement")


    def _prepare_symlink(self,):

        """
        进行 相关文件的软连接构建
        """

        if self.previous_file:
            for _file in self.previous_file:
                generate_symlink(_file.parent, self.task_dir, [_file.name])


    def _prepost(self):

        """
        预备的后处理
        """
        
        raise NotImplementedError("method not implement")

    def _post(self):

        """
        调用可能存在的后处理函数 post_func进行进一步后处理
        """

        self._prepost() #在这里调用预备的后处理

        if self.post_func:
            self.post_func()


    def prepare(self, return_submission_only=   False):

        """
        产生任务路径、产生化学结构、产生输入文件
        """

        create_path([self.task_dir])

        self._prepare_symlink()

        try: 
            if self.struct:
                self._dump_struct()
        except:
            pass

        if self.make_func:
            self._dump_input()
        self.prepare_tag = True

    def submit(self):
        """
        进行任务提交
        """
        
        raise NotImplementedError("method not implement")




class RunCP2K(RunBase):

    """
    继承RunBase基类，用来描述cp2k计算任务的过程， 包括cp2k任务输入文件的制作、任务的提交、任务的后处理过程
    """

    def __init__(self,

        work_base: str,
        task_dir,
        structure,
        make_func,
        disp_resource,
        previous_file:list =list(),
        post_func:FunctionType=None,
        cell=None,  
        mk_restart='./restart',
        input_name='input.inp',
        #force_field_dir=None,
        run_name="cp2k_untitiled",

        ):

        self.struct_name = "coord.xyz"

        super().__init__(work_base=work_base,
            task_dir=task_dir,
            make_func=make_func,
            previous_file=previous_file,
            disp_resource=disp_resource, 
            mk_restart=mk_restart,
            input_name=input_name,
            post_func=post_func,
            run_name=run_name,
            )

        self.set_structure(structure, cell=cell)

        self._prepare_task()


    def _prepare_task(self,**task_kwargs):

        """
        准备提交所需要的任务信息， 这里只是描述任务信息的字典
        """

        
        partask = partial(dict,
            task_work_path=str(self.rela_task_dir),
            command= f"mpiexec.hydra cp2k.popt -i {self.input_name} ",
            forward_files=[self.struct_name, f"{self.input_name}", f"{self.mk_restart}", "*"],
            backward_files= ["*output*",f"{self.mk_restart}", "*pos*","*frc*", "*-1.ener"],
            outlog="output.log",
            errlog= "cp2k.err",
        )
        partask = partial(partask, **task_kwargs)
        task = partask()

        self._task = task
        return task



    def _dump_struct(self):

        """
        输出结构文件
        """

        #2. 制作结构文件
        specorder=None
        print("dummy structure output")
        with open(self.task_dir/self.struct_name,"w") as f:
            
            f.write(self.make_func())
        pass
    
    def _dump_input(self):
        
        """
        产生计算的输入文件
        """

        self.input_txt = self.make_func()

        dump_input(self.input_txt, self.task_dir, input_name=self.input_name, restart_dir=self.mk_restart)

        pass



    def _prepost(self):

        """
        示例的处理cp2k的结果 进行一部分后处理
        """

        from dateutil.parser import parse as time_parse

        backward_files = self._task.backward_files
        tmp = list(filter(lambda x: "output" in x,backward_files))
        assert len(tmp) == 1
        tmp = list(Path(self.task_dir).glob(tmp[0]))
        assert len(tmp) == 1
        cp2k_output = tmp[0]
        print(tmp[0])

        with open(self.task_dir/(cp2k_output),"r") as f:
            line=True
            while line:
                line = f.readline()
                if "PROGRAM STARTED AT" in line:
                    time_str = line.split(" "*10)[-1]
                    start_time = time_parse(time_str)
                if "PROGRAM ENDED AT" in line:
                    time_str = line.split(" "*10)[-1]
                    end_time = time_parse(time_str)
        delta = str(end_time - start_time)
        with open(self.task_dir/("simple_report.txt"), "w") as f:

            f.write(f"total calculation time: {delta}")


    def submit(self):

        """
        进行任务提交
        """
        
        dummy_submit(self._task, self.submit_resource)

        print("dummy submit successfully running")
        #self._post()

        return True



class RunGroup():

    """
    用来聚集Run类的抽象，满足多个run类中包含的任务的统一提交
    """


    #aggregate runs to submit together
    # run_list will be the RunCP2K list

    
    def __init__(self, 
        run_list: list, 
        disp_resource,
        submission_info=None,
        auto_prepare=True,
    ):

        self.runmetas = run_list
        self._check_work_base()
        self._check_run_prepared()
        self._load_resource(disp_resource)

        self._generate_tasks()
        self.auto_prepare = auto_prepare


    def _generate_tasks(self):
        
        """
        取runmetas的run类包含的所有任务信息
        """

        self.tasks = [run._task for run in self.runmetas]

    def _load_resource(self, resource):

        """
        加载提交需要的计算配置信息
        """

        self.submit_resource = resource

    def _check_run_prepared(self):
        
        """
        检查run类中是否完成创建路径、输出需要的文件等步骤
        """

        tags = []
        for run in self.runmetas:
            tags.append(run.prepare_tag)

        self._all_prepared = all(tags)

        return self._all_prepared

    def _check_work_base(self):

        """
        检查包含的Run类中的work_base是否一致
        """

        tmp_base = self.runmetas[0].work_base
        for run in self.runmetas:
            assert run.work_base == tmp_base
        self.work_base = tmp_base

    def submit(self, **s_data):

        """
        将生成的task_list进行统一提交
        """

        if not self.auto_prepare:
            assert self._all_prepared == True
        else:
            for run in self.runmetas:  
                if run.prepare_tag == False:
                    run.prepare()
                    print("run runmeta have been prepared automatically.")
        
        dummy_submit(self,tasks, self.submit_resource)

        print("finished total dummy submission.")

        return True




class RunDPMD(RunBase):

    run_info = {
        "": "",
        "": "",}

    def __init__(self,

        work_base: str,
        task_dir: str,
        graph_dirs:list,
        structure: ase.Atoms,
        make_func: FunctionType,
        disp_resource: dict,

        previous_file:list =list(),
        cell=None,
        mk_restart: str or bool = './restart',
        input_name: str = 'input.lammps',
        forward_common_files: list=list() ,
        backward_common_files:list=list() ,
        
        run_name: str='dpmd_untitiled', 
        specorder: list=None,

        ):

        super().__init__(work_base=work_base,
            task_dir=task_dir,
            make_func=make_func,
            previous_file=previous_file,
            disp_resource=disp_resource, 
            mk_restart=mk_restart,
            input_name=input_name,
            forward_common_files=forward_common_files,
            backward_common_files=backward_common_files,
            run_name=run_name,
            
            )

        self.set_structure(structure, cell=cell)

        self.struct_name = "struct.lmps"

        abs_graph_dirs = []
        for _dir in graph_dirs:
            if os.path.isabs(_dir):
                try:
                    Path(_dir).relative_to(self.work_base)
                    abs_graph_dirs.append(Path(_dir))
                except:
                    raise Exception("graph_dir must relate to work_base dir!")
            else:
                abs_graph_dirs.append(Path(self.work_base/(_dir)))
        self.graph_dirs = abs_graph_dirs

        return_times = str(self.task_dir.relative_to(self.work_base)).count("/")+1
        rela_graph_dirs = []
        for _dir in abs_graph_dirs:
            _rela_graph_dir = Path("../"*return_times + str(Path(_dir).relative_to(self.work_base)))
            rela_graph_dirs.append(_rela_graph_dir)

        self.rela_graph_dirs = rela_graph_dirs

        self.specorder = specorder

        self._prepare_task()


    def _prepare_task(self,**task_kwargs):  #满足可能的不同需要，这样可以办到 但是好吗？
        
        
        partask = partial(Task,
            task_work_path=str(self.rela_task_dir),
            command= f"lmp_mpi -i {self.input_name}",
            forward_files=["*.lmps", "input.lammps", f"{self.mk_restart}"],
            backward_files= ["model_devi.out", "model_devi.log", "traj.xyz",f"{self.mk_restart}"],
            outlog="model_devi.log",
            errlog= "err.log",
        )
        partask = partial(partask, **task_kwargs)
        task = partask()

        self._task = task
        return task

    def _dump_struct(self):

        io.write(self.task_dir/(self.struct_name), self.struct, format="lammps-data", specorder=self.specorder)

    def _make_input_txt(self):

        self.input_txt = self.make_func()

    def _dump_input(self):

        dump_input(self.input_txt, self.task_dir, input_name=self.input_name, restart_dir=self.mk_restart)


    def _submit(self, forward_common_files=list(), backward_common_files=list(), **kwargs):

        machine = Machine.load_from_dict(self.submit_resource["machine"])
        resource = Resources.load_from_dict(self.submit_resource["resource"])
        sub = Submission(
            work_base=str(self.work_base), 
            machine=machine,
            resources=resource,
            task_list = [self._task],
            forward_common_files= [str(Path(_dir).relative_to(self.work_base)) for _dir in self.graph_dirs]+forward_common_files,
            backward_common_files= backward_common_files, 
            **kwargs,
        )

        sub.run_submission()

        return sub


class RunAnalysis(RunBase):


    def __init__(self, 

        work_base, 
        task_dir,
        ana_filename, #不同的分析情况，可能不一定需要ana_filename的需要
        analysis_func,
        previous_file,
        disp_resource,
        mk_restart='./restart',
        input_name="sub_script.py",
        post_func=None,
        run_name=None,
        ):

        super().__init__(work_base=work_base,
            task_dir=task_dir,
            make_func=analysis_func, 
            previous_file=previous_file,
            disp_resource=disp_resource, 
            mk_restart=mk_restart,
            input_name=input_name,
            post_func=post_func,
            run_name=run_name,
            )

        #self.ana_func = analysis_func
        self._render_ana_func = self._render_make_func
        self.ana_filename = ana_filename


        self._load_resource(disp_resource)

        self.run_type = 'local' #BUG

        self._prepare_task()

    @property
    def _final_make_func(self):

        return partial(make_ana_script_func, partial(self.make_func), self.ana_filename)

    @property
    def ana_func(self):
        
        return self.make_func


    def _load_resource(self, resource):

        self.submit_resource = resource


    def _make_input_txt(self):

        #3. 制作input 文件
        make_func = self._final_make_func
        self.input_txt = make_func()
        pass

    def _dump_struct(self):

        pass

    def _dump_input(self):
        
        if self.run_type != "local":
            dump_input(self.input_txt, self.task_dir, input_name=self.input_name, restart_dir=self.mk_restart)

        pass

    def _submit(self, **disp_sub_kwargs):

        if self.run_type == "local":
            ana_result = self._run_local()
        else:
            ana_result = self._run_disp(**disp_sub_kwargs) 
        
        self._prepost()
        if self.post_func:
            self.post_func() 

        return ana_result

    def _run_local(self):

        pre_dir = os.path.abspath(os.curdir)
        os.chdir(self.task_dir)
        result = self.ana_func() 
        os.chdir(pre_dir)
        
        return result

    def _run_disp(self, **kwargs):
        sub = Submission(
            work_base=str(self.work_base), 
            machine=self.submit_resource["machine"],
            resources=self.submit_resource["resource"],
            task_list = [self._task],
            forward_common_files= [blabla],
            backward_common_files= [gulugulu], 
            **kwargs,
        )

        sub.run_submission()

        return None


    def _prepost(self):

        pass


    def _prepare_task(self):

        
        task = None
        self._task = task

