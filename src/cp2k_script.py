from .run import *

run =RunCP2K("/home/username/trash","trash","abc",dummy_make_func, {"1":"2"})

run._render_make_func(structure="struct", cell=[1,2,3], basis_set=[1,2,],calc_type="cp2k")

run.prepare()
run.submit()
