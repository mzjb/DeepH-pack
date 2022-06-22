num_element = int(input("Number of atomic types: "))
atomic_number = []
num_orbitals = []
assert num_element > 0, "Number of atomic types should be greater than 0."
for index_element in range(num_element):
    input1 = int(input(f"Atomic type #{index_element + 1}'s atomic number: "))
    assert input1 > 0, "Atomic number should be greater than 0."
    input2 = int(input(f"Atomic type #{index_element + 1}'s orbiatl basis number: "))
    assert input2 > 0, "Orbiatl basis number should be greater than 0."
    atomic_number.append(input1)
    num_orbitals.append(input2)

orbital_str = '['
first_flag = True
for ele_i, ele_j in ((ele_i, ele_j) for ele_i in range(num_element) for ele_j in range(num_element)):
    for orb_i, orb_j in ((orb_i, orb_j) for orb_i in range(num_orbitals[ele_i]) for orb_j in range(num_orbitals[ele_j])):
        if first_flag:
            orbital_str += '{'
            first_flag = False
        else:
            orbital_str += ', {'
        orbital_str += f'"{atomic_number[ele_i]} {atomic_number[ele_j]}": [{orb_i}, {orb_j}]}}'
orbital_str += ']'
print("<orbital> keyword can be set as:")
print(orbital_str)
