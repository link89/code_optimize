#from .run import *

run_0 =RunCP2K("/home/kxiong/test","test_0","struct_info",dummy_make_func, {"machine":"a_remote_machine","resource":"cpu_and_gpu_usage"})
run_0._render_make_func(structure="struct", cell=[1,2,3], basis_set=[1,2,],calc_type="cp2k")
run_0.prepare()
run_0.submit()


run_1 =RunCP2K("/home/kxiong/test","test_1","struct_info",dummy_make_func, {"machine":"a_remote_machine","resource":"cpu_and_gpu_usage"})
run_2 =RunCP2K("/home/kxiong/test","test_2","struct_info",dummy_make_func, {"machine":"a_remote_machine","resource":"cpu_and_gpu_usage"})
run_3 =RunCP2K("/home/kxiong/test","test_3","struct_info",dummy_make_func, {"machine":"a_remote_machine","resource":"cpu_and_gpu_usage"})

run_1._render_make_func(structure="struct", cell=[1,2,3], basis_set=[1,2,],calc_type="cp2k")
run_2._render_make_func(structure="struct", cell=[1,2,3], basis_set=[1,2,],calc_type="cp2k")
run_3._render_make_func(structure="struct", cell=[1,2,3], basis_set=[1,2,],calc_type="cp2k")


rungroup = RunGroup(
    run_list=[run_0, run_1, run_2, run_3],
    disp_resource={"machine":"a_remote_machine","resource":"cpu_and_gpu_usage"})

rungroup.submit()