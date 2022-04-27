import os
import struct
import numpy as np
import re

from functools import partial
from pathlib import Path
import copy
from types import FunctionType


def create_path(path_list, parents=True, exit_ok=False):

    for path in path_list:
        Path(path).mkdir(parents=parents, exist_ok=exit_ok) 


def dump_input(text, task_path, input_name, restart_dir=False):

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



def dummy_submit(task, disp_resource):
    #mock function for uploading calculation task to HPC and retrieve the results.

    return True


def dummy_make_func(structure, cell, basis_set, calc_type):

    #mock function to generate a input file for science calculations.
    txt = f"{structure}\n{cell}\n{basis_set}\n{calc_type}"

    return txt

class RunBase():

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
        #检查task_dir路径相对还是绝对，最后用绝对的形式保存

        if os.path.isabs(new_task_dir):
            self.task_dir = Path(new_task_dir)
        else:
            self.task_dir = Path(self.work_base/(new_task_dir))

        print(self.task_dir)
        self.rela_task_dir = self.task_dir.relative_to(self.work_base)


    def set_structure(self, new_structure, cell=None):

        self.struct = new_structure
        if cell:
            self.struct.set_cell(cell)



    def _render_make_func(self, *make_args, **make_kwargs):

        self.make_func = partial(self.make_func,*make_args, **make_kwargs)

    def _reproduce(self, new_task_dir, **changed_make_kwargs):

        new_run= copy.deepcopy(self)
        new_run.task_dir = Path(new_task_dir)
        new_run._render_make_func(**changed_make_kwargs)

        return new_run

    def _check_make_function(self):

        try:
            self.make_func() 
        except:
            raise Exception("make function in RunMeta has not been prepared yet!, please check the make function.")
        pass

    def _check_resource():

        pass

    def _load_resource(self, resource):

        self.submit_resource = resource

    def _prepare_task(self, **task_kwargs):
        pass


    def _dump_struct(self):

        raise NotImplementedError("method not implement")
    
    def _dump_input(self):

        raise NotImplementedError("method not implement")


    def _prepare_symlink(self,):

        if self.previous_file:
            for _file in self.previous_file:
                generate_symlink(_file.parent, self.task_dir, [_file.name])


    def _prepost(self):
        
        raise NotImplementedError("method not implement")

    def _post(self):

        self._prepost()

        if self.post_func:
            self.post_func()


    def prepare(self, return_submission_only=   False):


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
        
        raise NotImplementedError("method not implement")




class RunCP2K(RunBase):

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

        #2. 制作结构文件
        specorder=None
        print("dummy structure output")
        with open(self.task_dir/self.struct_name,"w") as f:
            
            f.write(self.make_func())
        pass
    
    def _dump_input(self):


        self.input_txt = self.make_func()

        dump_input(self.input_txt, self.task_dir, input_name=self.input_name, restart_dir=self.mk_restart)

        pass



    def _prepost(self):

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
        
        dummy_submit(self._task, self.submit_resource)

        print("dummy submit successfully running")
        #self._post()

        return True



class RunGroup():

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

        self.tasks = [run._task for run in self.runmetas]

    def _load_resource(self, resource):

        self.submit_resource = resource

    def _check_run_prepared(self):
        
        tags = []
        for run in self.runmetas:
            tags.append(run.prepare_tag)

        self._all_prepared = all(tags)

        return self._all_prepared

    def _check_work_base(self):

        tmp_base = self.runmetas[0].work_base
        for run in self.runmetas:
            assert run.work_base == tmp_base
        self.work_base = tmp_base

    def submit(self, **s_data):

        if not self.auto_prepare:
            assert self._all_prepared == True
        else:
            for run in self.runmetas:  
                if run.prepare_tag == False:
                    run.prepare()
                    print("run runmeta have been prepared automatically.")

        print("finished total dummy submission.")

        return True