# data source: https://www.worldometers.info/coronavirus/country/uk/

import numpy as np
from matplotlib import pyplot as plt
import datetime as dt
from scipy.optimize import curve_fit
import matplotlib.patches as mpl_patches

# UK cases from '2020-02-27'
N_uk = 65382556

days_uk = np.arange('2020-02-27', '2020-04-21', dtype='datetime64[D]')

total_cases_per_day_uk = np.array([3, 4, 3, 13, 3, 12, 36, 29, 48, 45, 69, 43, 62,
                                   77, 130, 208, 342, 251, 152, 407, 676, 643, 714,
                                   1035, 665, 967, 1427, 1452, 2129, 2885, 2546,
                                   2433, 2619, 3009, 4324, 4244, 4450, 4735,
                                   5903, 3802, 3634, 5491, 4344, 8681, 5233, 5288,
                                   4342, 5252, 4603, 4617, 5599, 5525, 5850, 4676])

deaths_per_day_uk = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 1, 2, 2, 1,
                              10, 14, 20, 16, 33, 40, 33, 56, 48, 54, 87, 43,
                              115, 181, 260, 209, 180, 381, 563, 569, 684, 708,
                              621, 439, 786, 938, 881, 980, 917, 737, 717, 778,
                              761, 861, 847, 888, 596, 449])

recovery_per_day_uk = np.array([0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 2, 32, 13, 0, 0, 0, 28, 0, 42, 209, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0])

cases_per_day_uk = total_cases_per_day_uk - \
    deaths_per_day_uk - recovery_per_day_uk

# IT cases from '2020-02-20'
N_it = 60480000

days_it = np.arange('2020-02-20', '2020-04-21', dtype='datetime64[D]')

total_cases_per_day_it = np.array([1, 17, 58, 78, 72, 94, 147, 185, 234, 239, 573, 335,
                                   466, 587, 769, 778, 1247, 1492, 1797, 979, 2313,
                                   2651, 2547, 3497, 3590, 3233, 3526, 4207, 5322,
                                   5986, 6557, 5560, 4789, 5249, 5210, 6203, 5909,
                                   5974, 5217, 4050, 4053, 4782, 4668, 4585, 4805,
                                   4316, 3599, 3039, 3836, 4204, 3951, 4694, 4092,
                                   3153, 2972, 2667, 3786, 3493, 3491, 3047, 2256])

deaths_per_day_it = np.array([0, 0, 0, 0, 4, 4, 0, 5, 4, 8, 12, 11, 27, 28, 41,
                              49, 36, 133, 97, 168, 196, 189, 250, 175, 368, 349,
                              345, 475, 427, 627, 793, 651, 601, 743, 683, 712,
                              919, 889, 756, 812, 837, 727, 760, 766, 681, 525,
                              636, 604, 542, 610, 570, 619, 431, 566, 602, 578,
                              525, 575, 482, 433, 454])

recovery_per_day_it = np.array([0, 1, 1, 0, -1, 1, 1, 42, 1, 4, 33, 66, 11, 116,
                                138, 109, 66, 33, 102, 280, 41, 213, 181, 527,
                                369, 414, 192, 1084, 415, 689, 943, 952, 408,
                                894, 1036, 999, 589, 1434, 646, 1590, 1109,
                                1118, 1431, 1480, 1238, 819, 1022, 1555,
                                2099, 1979, 1985, 2079, 1677, 1224, 1695,
                                962, 2072, 2563, 2200, 2128, 1822])

cases_per_day_it = total_cases_per_day_it - \
    deaths_per_day_it - recovery_per_day_it


def moving_central_filter(a: np.ndarray) -> np.ndarray:

    n = len(a)
    hat_a = np.zeros(n)

    hat_a[0] = a[0]
    for k in range(1, n - 1):
        hat_a[k] = (a[k + 1] + a[k] + a[k - 1]) / 3.0
    hat_a[n - 1] = a[n - 1]

    return hat_a


def first_differential(a: np.ndarray) -> np.ndarray:

    n = len(a)
    da = np.zeros(n)

    da[0] = (-3.0 * a[0] + 4.0 * a[1] - a[2]) / 2.0
    for k in range(1, n - 1):
        da[k] = (a[k + 1] - a[k - 1]) / 2.0
    da[n - 1] = (3.0 * a[n - 1] - 4.0 * a[n - 2] + a[n - 3]) / 2.0

    return da


def second_differential(a: np.ndarray) -> np.ndarray:

    n = len(a)
    d2a = np.zeros(n)

    d2a[0] = 2.0 * a[0] - 5.0 * a[1] + 4.0 * a[2] - a[3]
    for k in range(1, n - 1):
        d2a[k] = a[k + 1] - 2.0 * a[k] + a[k + 1]
    d2a[n - 1] = 2.0 * a[n-1] - 5.0 * a[n-2] + 4.0 * a[n-3] - a[n-4]

    return d2a


def integrate_per_day(data_per_day: np.ndarray) -> np.ndarray:

    n = len(data_per_day)
    data = np.zeros(n)
    data[0] = data_per_day[0]
    for i in range(1, n):
        data[i] = data[i-1] + data_per_day[i]

    return data


def integrate_all(cases_per_day: np.ndarray = cases_per_day_uk,
                  deaths_per_day: np.ndarray = deaths_per_day_uk,
                  recovery_per_day: np.ndarray = recovery_per_day_uk) -> np.ndarray:

    total_cases = integrate_per_day(cases_per_day)
    total_deaths = integrate_per_day(deaths_per_day)
    total_recovery = integrate_per_day(recovery_per_day)

    return total_cases, total_deaths, total_recovery


def prepare_SIR_model(cases_per_day: np.ndarray = cases_per_day_uk,
                      deaths_per_day: np.ndarray = deaths_per_day_uk,
                      recovery_per_day: np.ndarray = recovery_per_day_uk,
                      total_population: int = N_uk, vital_dynamics: bool = True) -> np.ndarray:

    total_cases, total_deaths, total_recovery = integrate_all(
        cases_per_day, deaths_per_day, recovery_per_day)

    # is the cumulative?
    I = total_cases / total_population
    D = total_deaths / total_population
    R = D + total_recovery / total_population
    # is the dayly rate?
    I = cases_per_day / total_population
    D = deaths_per_day / total_population
    R = D + recovery_per_day / total_population

    I = moving_central_filter(I)
    R = moving_central_filter(R)
    S = 1.0 - I - R

    dI = first_differential(I)
    dR = first_differential(R)
    dS = - dI - dR

    if vital_dynamics:
        gamma_sc, lambda_sc, mu_sc = parameters_with_vital_dynamics(
            S, I, R, dS, dI, dR)
    else:
        gamma_sc, lambda_sc, mu_sc = parameters_without_vital_dynamics(
            S, I, R, dS, dI, dR)

    return I, R, S, gamma_sc, lambda_sc, mu_sc


def parameters_with_vital_dynamics(S: np.ndarray, I: np.ndarray, R: np.ndarray,
                                   dS: np.ndarray, dI: np.ndarray, dR: np.ndarray):

    n = len(I)

    lambda_field = np.zeros(n)
    gamma_field = np.zeros(n)
    mu_field = np.zeros(n)
    x_old = np.array([1, 1, 1])
    for k in range(n):
        b = np.array([dS[k], dI[k], dR[k]])
        A = np.array([[- S[k] * I[k], 0.0, I[k] + R[k]],
                      [S[k] * I[k], -I[k], -I[k]], [0.0, I[k], - R[k]]])
        A1 = np.matmul(A.T, A)
        b1 = np.matmul(A.T, b)
        try:
            x = np.linalg.solve(A1, b1)
            x_old = x
        except:
            x = x_old
        lambda_field[k] = x[0]
        gamma_field[k] = x[1]
        mu_field[k] = x[2]

    lambda_sc = abs(np.mean(np.ma.masked_invalid(lambda_field)))
    gamma_sc = abs(np.mean(np.ma.masked_invalid(gamma_field)))
    mu_sc = abs(np.mean(np.ma.masked_invalid(mu_field)))

    print(f"Preconditioned lambda = {lambda_sc}[1/day]")
    print(f"Preconditioned gamma = {gamma_sc}[1/day]")
    print(f"Preconditioned mu = {mu_sc}[1/day]")

    E = 0.0
    E = SIR_functional(S, I, R, dS, dI, dR, lambda_sc, gamma_sc, mu_sc, n)
    print(f"Initial Error = {E}[1/day]")

    b = np.array([0.0, 0.0, 0.0])
    A = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    b0 = 0.0
    for k in range(n):
        b0 += -(dS[k] - dI[k]) * S[k] * I[k]

    b[0] = b0/n

    b1 = 0.0
    for k in range(n):
        b1 += -(dI[k] - dR[k]) * I[k]

    b[1] = b1/n

    b2 = 0.0
    for k in range(n):
        b2 += -(dS[k]*(S[k] - 1.0) + dI[k] * I[k] + dR[k] * R[k])

    b[2] = b2/n

    A00 = 0.0
    for k in range(n):
        A00 += 2.0 * S[k] * S[k] * I[k] * I[k]

    A[0][0] = A00/n

    A01 = 0.0
    for k in range(n):
        A01 += - S[k] * I[k] * I[k]

    A[0][1] = A01/n
    A[1][0] = A01/n

    A02 = 0.0
    for k in range(n):
        A02 += (S[k] - 1.0 - I[k]) * S[k] * I[k]

    A[0][2] = A02/n
    A[2][0] = A02/n

    A11 = 0.0
    for k in range(n):
        A11 += 2.0 * I[k] * I[k]

    A12 = 0.0
    for k in range(n):
        A12 += (I[k] - R[k]) * I[k]

    A[1][2] = A12/n
    A[2][1] = A12/n

    A22 = 0.0
    for k in range(n):
        A22 += ((S[k] - 1.0) * (S[k] - 1.0) + I[k] * I[k] + R[k] * R[k])

    A[2][2] = A22/n

    x = np.linalg.solve(A, b)
    if x[0] > 0.0:
        lambda_sc = x[0]
        print(f"Best Fit lambda = {lambda_sc}[1/day]")
    else:
        print(f"Negative lambda, use preconditioned value.")

    if x[1] > 0.0:
        gamma_sc = x[1]
        print(f"Best Fit gamma = {gamma_sc}[1/day]")
    else:
        print(f"Negative gamma, use preconditioned value.")

    if x[2] > 0.0:
        mu_sc = x[2]
        print(f"Best Fit mu = {mu_sc}[1/day]")
    else:
        print(f"Negative mu, use preconditioned value.")

    E = SIR_functional(S, I, R, dS, dI, dR, lambda_sc, gamma_sc, mu_sc, n)
    print(f"Final Error = {E}[1/day]")

    '''                   
    dEdlambda = 0.0
    for k in range(n):
        dEdlambda += (dS[k] - dI[k]) * S[k] * I[k] \
            + 2.0 * lambda_sc * S[k] * S[k] * I[k] * I[k] \
                - gamma_sc * I[k] * S[k] * I[k] \
                    + mu_sc * (S[k] - 1.0 - I[k]) * S[k] * I[k]
         
    dEdgamma = 0.0
    for k in range(n):
        dEdgamma += (dI[k] - dR[k]) * I[k] \
            - lambda_sc * S[k] * I[k] * I[k] \
                + 2.0 * gamma_sc * I[k] * I[k] \
                    + mu_sc * (I[k] - R[k]) * I[k]
                
    dEdmu = 0.0
    for k in range(n):
        dEdmu += (dS[k]*(S[k] - 1.0) + dI[k] * I[k] + dR[k]* R[k]) \
            + lambda_sc * (S[k] - 1.0 - I[k]) * S[k] * I[k] \
                + gamma_sc * I[k] * (I[k] - R[k]) \
                    + mu_sc * ((S[k] - 1.0) * (S[k] - 1.0) + I[k] * I[k] + R[k] * R[k])
    '''

    return gamma_sc, lambda_sc, mu_sc


def parameters_without_vital_dynamics(S: np.ndarray, I: np.ndarray, R: np.ndarray,
                                      dS: np.ndarray, dI: np.ndarray, dR: np.ndarray):

    mu_sc = 0.0

    gamma_field = (dR + mu_sc * R) / I
    gamma_sc = np.mean(gamma_field)

    lambda_field = (dI + (gamma_sc + mu_sc) * I) / (S * I)
    lambda_sc = np.mean(lambda_field)

    print(f"Preconditioned lambda = {lambda_sc}[1/day]")
    print(f"Preconditioned gamma = {gamma_sc}[1/day]")

    n = len(I)

    E = SIR_functional(S, I, R, dS, dI, dR, lambda_sc, gamma_sc, mu_sc, n)
    print(f"Initial Error = {E}[1/day]")

    b = np.array([0.0, 0.0])
    A = np.array([[0.0, 0.0], [0.0, 0.0]])

    b0 = 0.0
    for k in range(n):
        b0 += -(dS[k] - dI[k]) * S[k] * I[k]

    b[0] = b0/n

    b1 = 0.0
    for k in range(n):
        b1 += -(dI[k] - dR[k]) * I[k]

    b[1] = b1/n

    A00 = 0.0
    for k in range(n):
        A00 += 2.0 * S[k] * S[k] * I[k] * I[k]

    A[0][0] = A00/n

    A01 = 0.0
    for k in range(n):
        A01 += - S[k] * I[k] * I[k]

    A[0][1] = A01/n
    A[1][0] = A01/n

    A11 = 0.0
    for k in range(n):
        A11 += 2.0 * I[k] * I[k]

    A[1][1] = A11/n

    x = np.linalg.solve(A, b)
    if x[0] > 0.0:
        lambda_sc = x[0]
        print(f"Best Fit lambda = {lambda_sc}[1/day]")
    else:
        print(f"Negative lambda, use preconditioned value.")

    if x[1] > 0.0:
        gamma_sc = x[1]
        print(f"Best Fit gamma = {gamma_sc}[1/day]")
    else:
        print(f"Negative gamma, use preconditioned value.")

    E = SIR_functional(S, I, R, dS, dI, dR, lambda_sc, gamma_sc, mu_sc, n)
    print(f"Final Error = {E}[1/day]")

    '''                
    dEdlambda = 0.0
    for k in range(n):
        dEdlambda += (dS[k] - dI[k]) * S[k] * I[k] \
            + 2.0 * lambda_sc * S[k] * S[k] * I[k] * I[k] \
                - gamma_sc * I[k] * S[k] * I[k]
         
    dEdgamma = 0.0
    for k in range(n):
        dEdgamma += (dI[k] - dR[k]) * I[k] \
            - lambda_sc * S[k] * I[k] * I[k] \
                + 2.0 * gamma_sc * I[k] * I[k]
    '''

    return gamma_sc, lambda_sc, mu_sc


def SIR_functional(S: np.ndarray, I: np.ndarray, R: np.ndarray,
                   dS: np.ndarray, dI: np.ndarray, dR: np.ndarray,
                   lambda_sc: float, gamma_sc: float, mu_sc: float, n: int):

    E = 0.0
    for k in range(n):
        E += 0.5 * (dS[k] + lambda_sc * S[k] * I[k] - mu_sc + mu_sc * S[k])**2.0 + \
            0.5 * (dI[k] - lambda_sc * S[k] * I[k] + (gamma_sc + mu_sc) * I[k])**2.0 + \
            0.5 * (dR[k] - gamma_sc * I[k] + mu_sc * R[k])**2.0

    return E


def SIR_model(I: float, R: float, gamma_sc: float, lambda_sc: float, mu_sc: float) -> float:

    S = 1.0 - I - R
    dIdt = lambda_sc * S * I - (gamma_sc + mu_sc) * I
    dRdt = gamma_sc * I - mu_sc * R
    # at equilibrium:
    # S_eq = (gamma_sc + mu_sc)/lambda_sc
    # I_eq = mu_sc * (lambda_sc - gamma_sc - mu_sc)/(lambda_sc * (gamma_sc + mu_sc) )
    # R_eq = gamma_sc * (lambda_sc - gamma_sc - mu_sc)/(lambda_sc * (gamma_sc + mu_sc) )

    return S, dIdt, dRdt


def runge_kutta_4_SIR(I_old: float, R_old: float, time_step: float, gamma_sc: float, lambda_sc: float, mu_sc: float) -> float:

    # step 1
    S_old, dIdt_1, dRdt_1 = SIR_model(I_old, R_old, gamma_sc, lambda_sc, mu_sc)
    I_temp = I_old + (time_step/2.0) * dIdt_1
    R_temp = R_old + (time_step/2.0) * dRdt_1

    # step 2
    S_temp, dIdt_2, dRdt_2 = SIR_model(
        I_temp, R_temp, gamma_sc, lambda_sc, mu_sc)
    I_temp = I_old + (time_step/2.0) * dIdt_2
    R_temp = R_old + (time_step/2.0) * dRdt_2

    # step 3
    S_temp, dIdt_3, dRdt_3 = SIR_model(
        I_temp, R_temp, gamma_sc, lambda_sc, mu_sc)
    I_temp = I_old + time_step * dIdt_3
    R_temp = R_old + time_step * dRdt_3

    # step 4
    S_temp, dIdt_4, dRdt_4 = SIR_model(
        I_temp, R_temp, gamma_sc, lambda_sc, mu_sc)
    I_new = I_old + (time_step / 6.0) * (dIdt_1 + 2.0 *
                                         dIdt_2 + 2.0 * dIdt_3 + dIdt_4)
    R_new = R_old + (time_step / 6.0) * (dRdt_1 + 2.0 *
                                         dRdt_2 + 2.0 * dRdt_3 + dRdt_4)
    S_new = 1.0 - I_new - R_new

    return I_new, R_new, S_new


def run_SIR_model(step_number: int, time_step: int = 1.0,
                  country: str = 'uk',
                  offset: int = 0,
                  vital_dynamics: bool = True) -> np.ndarray:

    cases_per_day, deaths_per_day, recovery_per_day, total_cases_per_day, total_population = input_arrays(
        country, offset)

    Id, Rd, Sd, gamma_sc, lambda_sc, mu_sc = prepare_SIR_model(
        cases_per_day, deaths_per_day, recovery_per_day, total_population,
        vital_dynamics)

    I = np.zeros(step_number)
    R = np.zeros(step_number)
    S = np.zeros(step_number)

    I[0] = Id[0]
    R[0] = Rd[0]
    S[0] = Sd[0]
    for k in range(1, step_number):

        I[k], R[k], S[k] = runge_kutta_4_SIR(
            I[k-1], R[k-1], time_step, gamma_sc, lambda_sc, mu_sc)

    total_cases, total_deaths, total_recovery = integrate_all(
        cases_per_day, deaths_per_day, recovery_per_day)

    # , total_cases, (total_deaths + total_recovery)
    return I * total_population, R * total_population, cases_per_day, (deaths_per_day + recovery_per_day)


def plot_results(Im: np.ndarray, Rm: np.ndarray, Id: np.ndarray, Rd: np.ndarray, country: str = 'uk'):

    _, _, _, _, total_population = input_arrays(country, 0)

    nm = len(Im)
    nd = len(Id)

    days, index = days_for_ticks(nd, nm, country)

    xm = np.linspace(0, nm-1, nm)
    xd = np.linspace(0, nd-1, nd)

    fig = plt.figure()
    plt.semilogy(xm, Im, 'r', label='model')
    plt.semilogy(xd, Id, 'o', label='data')
    plt.semilogy(xm, total_population * np.ones(nm),
                 '--k', label='total population')
    plt.legend(loc='best')
    # plt.xlabel('days since beginning')
    plt.ylabel('Total Infected')
    plt.title('SIR model with Vital Dynamics (UK)')
    plt.grid()
    plt.xticks(index, days, rotation=45)
    fig.savefig(f'fig1.png')

    fig = plt.figure()
    plt.semilogy(xm, Rm, 'r', label='model')
    plt.semilogy(xd, Rd, 'o', label='data')
    plt.semilogy(xm, total_population * np.ones(nm),
                 '--k', label='total population')
    plt.legend(loc='best')
    # plt.xlabel('days since beginning')
    plt.ylabel('Total Outcomes')
    plt.title('SIR model with Vital Dynamics (UK)')
    plt.grid()
    # UK
    plt.xticks(index, days, rotation=45)
    fig.savefig(f'fig2.png')

    return


def input_arrays(country: str = 'uk', offset: int = 0):

    if country == 'uk':
        cases_per_day = cases_per_day_uk[offset:]
        deaths_per_day = deaths_per_day_uk[offset:]
        recovery_per_day = recovery_per_day_uk[offset:]
        total_population = N_uk
    elif country == 'it':
        cases_per_day = cases_per_day_it[offset:]
        deaths_per_day = deaths_per_day_it[offset:]
        recovery_per_day = recovery_per_day_it[offset:]
        total_population = N_it
    else:
        print(f"Data for this country not stored. Abort")
        return

    total_cases_per_day = cases_per_day + deaths_per_day + recovery_per_day

    return cases_per_day, deaths_per_day, recovery_per_day, total_cases_per_day, total_population


def input_days(length: int, country: str = 'uk'):

    if country == 'uk':
        days = days_uk
    elif country == 'it':
        days = days_it
    else:
        print(f"Data for this country not stored. Abort")
        return

    n_days = len(days)
    m = n_days - length
    days = days[m:]

    return days


def gaus(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def linear(x, a, b):
    return a*x + b


def covid19_gauss(country: str = 'uk', offset: int = 0):

    cases_per_day, deaths_per_day, recovery_per_day, total_cases_per_day, total_population = input_arrays(
        country, offset)

    n = len(cases_per_day)
    x = np.linspace(0, n-1, n)
    m = n+15
    x2 = np.linspace(0, m-1, m)

    fit_gauss_with_figure(cases_per_day, x, x2,
                          'New Cases per Day', 'new cases', country)

    fit_gauss_with_figure(deaths_per_day, x, x2,
                          'Deaths per Day', 'deaths', country)

    fit_gauss_with_figure(recovery_per_day, x, x2,
                          'Recovery per Day', 'recovered', country)

    fit_gauss_with_figure(total_cases_per_day, x, x2,
                          'Total Cases per Day', 'total', country)


def fit_gauss_with_figure(y: np.ndarray, x: np.ndarray, x2: np.ndarray, title_str: str, y_str: str, country: str):

    nd = len(y)
    n2 = len(x2)
    days, index = days_for_ticks(nd, n2, country)

    mean = np.mean(y)
    sigma = (np.var(y))**0.5

    popt, pcov = curve_fit(gaus, x, y, p0=[1, mean, sigma])

    plt.semilogy(x, y, 'b+:', label='data')
    plt.semilogy(x2, gaus(x2, *popt), 'ro:', label='fit')
    plt.legend()
    plt.title(title_str+' in '+country)
    plt.xlabel('days')
    plt.ylabel(y_str)
    plt.xticks(index, days, rotation=45)
    plt.grid()
    plt.show()

    return


def correlations(country: str = 'uk', offset: int = 0):

    cases_per_day, deaths_per_day, recovery_per_day, total_cases_per_day, total_population = input_arrays(
        country, offset)

    I, R, S, _, _, _ = prepare_SIR_model(
        cases_per_day, deaths_per_day, recovery_per_day, total_population,
        False)

    I = moving_central_filter(I)
    R = moving_central_filter(R)
    S = 1.0 - I - R

    dI = first_differential(I)
    dR = first_differential(R)
    dS = - dI - dR

    handles = [mpl_patches.Rectangle(
        (0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)]

    C = np.corrcoef(np.ma.masked_invalid(I*I), np.ma.masked_invalid(dI))
    rho = C[0][1]

    popt, pcov = curve_fit(linear, I*I, dI, p0=[1, 1])

    fig = plt.figure()
    plt.plot(I*I, dI, 'or')
    plt.plot(I*I, linear(I*I, *popt))
    plt.xlabel('I^2')
    plt.ylabel('dI/dt [1/day]')
    labels = []
    labels.append(f"rho = {round(rho, 4)}")
    plt.legend(handles, labels, loc='best', fontsize='medium',
               fancybox=False, framealpha=0,
               handlelength=0, handletextpad=0)
    plt.title('dI/dt vs I^2')
    fig.savefig(f'fig_1.png')

    C = np.corrcoef(np.ma.masked_invalid(I*R), np.ma.masked_invalid(dI))
    rho = C[0][1]

    popt, pcov = curve_fit(linear, I*R, dI, p0=[1, 1])

    fig = plt.figure()
    plt.plot(I*R, dI, 'or')
    plt.plot(I*R, linear(I*R, *popt))
    plt.xlabel('IR')
    plt.ylabel('dI/dt [1/day]')
    labels = []
    labels.append(f"rho = {round(rho, 4)}")
    plt.legend(handles, labels, loc='best', fontsize='medium',
               fancybox=False, framealpha=0,
               handlelength=0, handletextpad=0)
    plt.title('dI/dt vs IR')
    fig.savefig(f'fig_2.png')

    C = np.corrcoef(np.ma.masked_invalid(R*R), np.ma.masked_invalid(dI))
    rho = C[0][1]

    popt, pcov = curve_fit(linear, R*R, dI, p0=[1, 1])

    fig = plt.figure()
    plt.plot(R*R, dI, 'or')
    plt.plot(R*R, linear(R*R, *popt))
    plt.xlabel('R^2')
    plt.ylabel('dI/dt [1/day]')
    labels = []
    labels.append(f"rho = {round(rho, 4)}")
    plt.legend(handles, labels, loc='best', fontsize='medium',
               fancybox=False, framealpha=0,
               handlelength=0, handletextpad=0)
    plt.title('dI/dt vs R^2')
    fig.savefig(f'fig_3.png')

    C = np.corrcoef(np.ma.masked_invalid(I), np.ma.masked_invalid(dI))
    rho = C[0][1]

    popt, pcov = curve_fit(linear, I, dI, p0=[1, 1])

    fig = plt.figure()
    plt.plot(I, dI, 'or')
    plt.plot(I, linear(I, *popt))
    plt.xlabel('I')
    plt.ylabel('dI/dt [1/day]')
    labels = []
    labels.append(f"rho = {round(rho, 4)}")
    plt.legend(handles, labels, loc='best', fontsize='medium',
               fancybox=False, framealpha=0,
               handlelength=0, handletextpad=0)
    plt.title('dI/dt vs I')
    fig.savefig(f'fig_4.png')

    C = np.corrcoef(np.ma.masked_invalid(R), np.ma.masked_invalid(dI))
    rho = C[0][1]

    popt, pcov = curve_fit(linear, I, dI, p0=[1, 1])
    fig = plt.figure()
    plt.plot(R, dI, 'or')
    plt.plot(R, linear(R, *popt))
    plt.xlabel('R')
    plt.ylabel('dI/dt [1/day]')
    labels = []
    labels.append(f"rho = {round(rho, 4)}")
    plt.legend(handles, labels, loc='best', fontsize='medium',
               fancybox=False, framealpha=0,
               handlelength=0, handletextpad=0)
    plt.title('dI/dt vs R')
    fig.savefig(f'fig_5.png')

    return


def days_for_ticks(nd: int, nm: int, country: str):

    days_d = input_days(nd, country)
    end_date = np.datetime64(days_d[0]) + np.timedelta64(nm, 'D')
    days2 = np.arange(days_d[0], end_date, dtype='datetime64[D]')
    days = days2[::7]
    index = np.linspace(0, 7*(len(days)-1), len(days))

    return days, index
