import numpy as np


def create_array(dim_1, dim_2):
    ar = np.random.rand(dim_1, dim_2)
    print(ar)


def solve_eqs(coeff_arr, solve_arr):
    return np.linalg.solve(coeff_arr, solve_arr)


# Find sensitivity of change in a coeffecient in the solution
def find_sensitivity(coeff_arr, solve_arr, dim_1, dim_2, percentage, bIncrease):
    sol = solve_eqs(coeff_arr, solve_arr)
    coeff_arr_copy = np.copy(coeff_arr)
    change = coeff_arr_copy[dim_1, dim_2] * (percentage / 100)
    if bIncrease:
        coeff_arr_copy[dim_1, dim_2] =   coeff_arr_copy[dim_1, dim_2]  + change
    else:
        coeff_arr_copy[dim_1, dim_2] =   coeff_arr_copy[dim_1, dim_2] - change

    sol_after_change = solve_eqs(coeff_arr_copy, solve_arr)
    ret_arr = ((sol_after_change - sol) / sol) * 100
    return ret_arr


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    create_array(2, 3)

    # SOLVE THE SYSTEM OF LINEAR EQUATIONS

    coeff_arr = np.array([[20.0, 4.0, 4.0], [10.0, 14.0, 5.0], [5.0, 5.0, 12.0]])
    solve_arr = np.array([6600.0, 5100.0, 3100.0])
    sol_arr = solve_eqs(coeff_arr, solve_arr)

    # If demand doubles what will be the solution
    solve_arr *= 2
    sol_arr1 = solve_eqs(coeff_arr, solve_arr)
    # As expected the solution vector has doubled

    # Check whether solution is correct
    # A.x = b
    ret = np.allclose(np.dot(coeff_arr, sol_arr1), solve_arr)

    # Find determinant of coeffecient matrix
    d_coeff = np.linalg.det(coeff_arr)

    # Determinant of the matrix is non zero that means all equations are linearly
    # independent and hence only one solution. Does not have free variables

    solve_arr1 = np.array([2000.0, 4000.0, 4000.0])
    sol_arr3 = solve_eqs(coeff_arr, solve_arr1)
    print(sol_arr3)

    solve_arr2 = solve_arr + solve_arr1
    sol_arr4 = solve_eqs(coeff_arr, solve_arr2)
    print(sol_arr4)
    # This solution is vector sum of the previous 2 solution
    # vectors i.e sol_arr4 = sol_arr + sol_arr3

    # SENSITIVITY AND ROBUSTNESS
    # Change the (0,0) element by 3% increase
    ret = find_sensitivity(coeff_arr, solve_arr, 0, 0, 3, True)
    print("Effect on solution when coeff (0,0) is increased by 3%")
    print(ret)

    ret = find_sensitivity(coeff_arr, solve_arr, 1, 1, 3, False)
    print("Effect on solution when coeff (1,1) in decreased by 3%")
    print(ret)

    ret = find_sensitivity(coeff_arr, solve_arr, 2, 1, 3, True)
    print("Effect on solution when coeff(2,1) is increased by 3%")
    print(ret)

    # A PLANT OFFLINE
    # Using first two equations you can solve for first two products
    # Find based on this solution what is left for the third product

    coeff_arr_2 = np.array([[20.0, 4.0], [10.0, 14.0]])
    solve_arr_2 = np.array([6600.0, 5100.0])
    sol_arr_2 = solve_eqs(coeff_arr_2, solve_arr_2)
    print(sol_arr_2)
    balance_qty_3rdprod = 3100.0 - ((sol_arr_2[0] * 5) + (sol_arr_2[1]*5))
    print("Balance amount of Gasoline to produced {0}".format(str(balance_qty_3rdprod)))
    extra_crude_Bfactory = (balance_qty_3rdprod/5.0)
    excess_Diesel_produced = extra_crude_Bfactory * 14.0
    excess_Motoroil_produced = extra_crude_Bfactory * 4
    print("Extra crude for B factory to produce balance Gasoline {0}".format(str(extra_crude_Bfactory)))
    print("Excess Motor Oil Produced {0} and Diesel produced {1}".format(
        excess_Motoroil_produced,
        excess_Diesel_produced
    ))



