import numpy as np
import sympy
import sympy as smp
from sympy.abc import x,y,z
import math as m

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

    # Use method of least square since the system of equations is
    # over determined. This does not provide an exact solution but
    # provide good approximation. In this case there is excessive
    # production for first two products and less production in third

    coeff_arr_2 = np.array([[20.0, 4.0], [10.0, 14.0], [5.0,5.0]])
    solve_arr_2 = np.array([6600.0, 5100.0, 3100.00])
    sol_arr_2 = np.linalg.lstsq(coeff_arr_2, solve_arr_2)
    print(sol_arr_2)

    # BUYING ANOTHER PLANT
    w = sympy.Symbol("w")
    A_aug = sympy.Matrix([[20, 4, 4, 5000-4*w],
                          [10, 14, 5, 8500-5*w],
                          [5, 5, 12, 10000-12*w]])
    # Row Reduced Echelon Form
    ret = A_aug.rref()
    # Solve for system of equations
    sol_arr = smp.solvers.solvers.solve_linear_system_LU(ret[0], [x, y, z])
    # Give value to symbol w. Since system of equations is underdetermined w becomes the free variable
    print("Production in first {0} second {1} third {2} fourth {3}".format(sol_arr[x].subs(w, 200), sol_arr[y].subs(w, 200),
                                                                           sol_arr[z].subs(w, 200), 200))
    # CALCULATE THE PARAFFIN SUPPLIED

    paraf_func = 3 * x + 5 * y + 2 * (z + w)
    print("Paraffin supplied is {0} ".format(paraf_func.subs(
        [(x, sol_arr[x].subs(w, 200)),
         (y, sol_arr[y].subs(w, 200)),
         (z, sol_arr[z].subs(w, 200)),
         (w, 200)]
    )))

    # SELLING FIRST PLANT

    # In this case system is overdetermined and does not have a
    # solution. We can use least square to find optimal approx
    # solution

    coeff_arr = np.array([[4.0, 4.0], [14.0, 5.0], [5.0, 12.0]])
    solve_arr = np.array([5000.0, 8500.0, 10000.0])
    sol_arr = np.linalg.lstsq(coeff_arr, solve_arr)
    print("After selling first plant optimal crude consumption for second {0} third {1}".format(
        sol_arr[0][0], sol_arr[0][1]
    ))

    w = sympy.Symbol("w")
    A_aug = sympy.Matrix([[4, 4, 5000-4*w],
                          [14, 5, 8500-5*w],
                          [5, 12, 10000-12*w]])
    ret = A_aug.rref()
    # Discuss on this. w related vector and z related vector are exactly the same hence this system is similar to system
    # above which is overdetermined??
    A_aug = sympy.Matrix([[4, 4, 6000-4*w],
                          [14, 5, 5100-5*w],
                          [5, 12, 3100-12*w]])
    ret1 = A_aug.rref()

    print("Done")

    # SET RATES FOR PRODUCTS
    M = np.matrix('20 10 5; 4 14 5; 4 5 12')
    R = np.matrix('100 104 102 96; 66 64 68 68; 102 100 98 100')
    ret = M * R
    max_val = ret.max()
    dict = {'fact':{0: 'A', 1: 'B', 2:'C'},
            'price':{0: 'First Option',
                     1: 'Second Option',
                     2: 'Third Option',
                     3: 'Fourth Option'}}

    [r, c] = np.where(ret == max_val)
    print("Best option Factory: {0} Pricing:{1}".format(
        dict['fact'][r[0]],
        dict['price'][c[0]]
    ))


    # MARGINAL COST
    cost_func = (0.005 * x * x * x ) - \
                (0.02 * x * x ) + (30 * x) + 5000

    cost_func_diff = smp.diff(cost_func, x)
    print(" Marginal Cost for 22 gallons is {0}".format(
        cost_func_diff.subs([(x, 22.0)])
    ))



    # MARGINAL REVENUE

    rev_func = (3.0 * x * x) + (36 * x) + 5
    rev_func_diff = smp.diff(rev_func, x)
    print(" Marginal Rev for 28 gallons sold {0}".format(
        rev_func_diff.subs([(x, 28.0)])
    ))

    # POUR CRUDE OIL IN TANK
    Ri = smp.Symbol('Ri')
    rad = smp.Symbol('rad')
    t = smp.Symbol('t')
    height_func = (Ri * t)/(m.pi * rad * rad)
    height_change_rate = smp.diff(height_func, t)

    print("Height after 2 hours {0}".format(
        height_change_rate.subs(
            [(Ri, 314),
             (rad, 10),
             (t, 2.0) # Eventhough not there just wanted to check
            ]
        ) * 2.0
    ))
    











