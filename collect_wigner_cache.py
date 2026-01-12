import sys
from wigner.su2_group import Su2Group

dim_limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

dim = 2
while True:
    print(dim)

    Su2Group(dim).kernel
    Su2Group(dim).wigner_basis
    Su2Group(dim).wigner_transform_expr

    if dim_limit and dim >= dim_limit:
        break

    dim += 1
