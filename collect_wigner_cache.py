from wigner.su2_group import Su2Group

dim = 2
while True:
    print(dim)

    Su2Group(dim).kernel
    Su2Group(dim).wigner_basis
    Su2Group(dim).wigner_transform_expr

    dim += 1
