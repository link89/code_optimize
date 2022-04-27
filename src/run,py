

def dummy_submit(task, disp_resource):
    #mock function for uploading calculation task to HPC and retrieve the results.

    return True


def dummy_make_func(structure, cell, basis_set, calc_type):

    #a function to generate a input file for science calculations.
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


    @property
    def task(self):

        return {"cp2k": "a dummy cp2k task"}


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

    def _make_input_txt(self):

        raise NotImplementedError("method not implement")

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
            self._make_input_txt()
            self._dump_input()
        self.prepare_tag = True

    def submit(self):
        
        dummy_submit(self.task, self.disp_resource)

        self._post()

        return True


        