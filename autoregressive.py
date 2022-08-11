import scipy as sp
from scipy import special
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import signal
import lmfit as lf
from numba import jit, prange

##################################################

font = {'family': 'serif',
        'weight': 'normal',
        'size': 9,
        }
ax_font = {'titlesize': 24,
           'labelsize': 18,
           }

mpl.rc('font', **font)
mpl.rc('axes', **ax_font)
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['mathtext.fontset'] = 'cm'

inter = 'lanczos'

##################################################

L = 10000    # 20000?
N = 8       # 16?
t = np.arange(L)


def autocor(sig, weirdnorm=False):
    arr = sig.copy()
    arr -= np.mean(arr)
    if weirdnorm:
        return(
            signal.correlate(arr, arr, mode='full')[len(arr)-1:-1] / len(arr)
        )
    return(
        signal.correlate(arr, arr, mode='full')[len(arr)-1:-1] /
        np.arange(len(arr), 1, -1)
    )


def cxx(t, b, etime=None, mrt=None):
    if mrt == None and etime == None:
        print("you need to set one of them")
        return()
    elif mrt != None and etime != None:
        print("only set one of them")
        return()
    elif etime != None:
        t0 = 10**etime / (special.gammainccinv(b, np.e**-1))**b
    else:
        t0 = 10**mrt / special.gamma(b+1)
    return np.exp(-(t/t0)**(1/b))


def exact(t, b, etime=None, mrt=None):
    if mrt == None and etime == None:
        print("you need to set one of them")
        return()
    elif mrt != None and etime != None:
        print("only set one of them")
        return()
    elif etime != None:
        t0 = 10**etime / (special.gammainccinv(b, np.e**-1))**b
    else:
        t0 = 10**mrt / special.gamma(b+1)
    return t0*special.gamma(b+1) * special.gammainc(b, (t/t0)**(1/b))


# def s_integrate(arr, axis=-1):
#     out = np.zeros_like(arr)
#     for i in range(1, L):
#         out[i-1] = integrate.simpson(arr[:i], axis=axis)
#     return out


@jit
def ma(ht, L=L):
    r = np.random.normal(0, 1, size=L)
    print(np.mean(r), np.std(r))
    r -= np.mean(r)
    r /= np.std(r)
    x = np.zeros(L)
    # print(len(r), len(ht))
    for i in range(L):
        #     if i > -1:
        #         print(i,
        #               len(r[i+1:min(L, i+L//2)]),
        #               len(ht[1:]))
        #     x[i] = np.sum(r[i:max(0, i-L//2):-1]*ht[:i]) + np.sum(r[i+1:min(L, i+L//2)] * ht[1:min(L//2, L-i)])
        # hey, if r and ht are the same length:
        x[i] = np.sum(r[i::-1]*ht[:i+1]) + np.sum(r[i+1:] * ht[1:L-i])
    return x


@jit(forceobj=True)
def lognormal_normal(stats):
    out = np.array(stats.copy())
    m, s = stats
    out[0] = 10**(m+s**2*np.log(10)**2/2)
    out[1] = (
        10**(2*m + s**2*np.log(10)) * (10**(s**2*np.log(10)) - 1)
    )**0.5
    return np.array(out)


def cumsimps(f):
    if f.ndim > 1:
        raise Exception('f must be a 1d array')
    out = np.zeros_like(f)
    out[0] = 0
    aa = 1/12 * (5*f[0] + 8*f[1] - f[2])
    bb = 1/24*(9*f[0] + 19*f[1] - 5*f[2] + f[3])
    cc = 1/720*(251*f[0] + 646*f[1] - 264*f[2] + 106*f[3] - 19*f[4])

    out[1] = np.min([aa, bb, cc])

    for n in range(2, len(f)):
        out[n] = out[n-2] + 1/3 * (f[n-2] + 4*f[n-1] + f[n])
    return out

##################################################


num_b = 15
# tlist = np.log10(np.array([0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]))
tlist = np.linspace(0, np.log10(L), 16)
estimates = np.zeros((num_b, len(tlist), 2, 2))


for b in range(1, 1+num_b):
    for i, thetime in enumerate(tlist):

        ct = cxx(t, b, thetime)
        # plt.plot(t, ct)
        # plt.xlim(0,100)
        # print(ct[0])

        sw = np.abs(sp.fft.hfft(ct))
        # sw -= np.min(sw)
        # print(sw[0])
        # plt.yscale('log')
        # plt.xscale("log")
        # plt.show()

        ht = np.abs(sp.fft.ihfft(np.sqrt(sw)))  # [:L//2]
        print(ht[0])
        ht /= ht[0]
        # ht[0] = 1
        # ht /= np.sum(ht**2)

        x = np.zeros((N, L))
        cx = np.zeros((N, L-1))
        icx = np.zeros_like(cx)

        for n in prange(N):
            # print(n)
            x[n] = ma(ht)
            x[n] -= np.mean(x[n])
            x[n] /= np.std(x[n])
            cx[n] = autocor(x[n])
            # icx[n] = s_integrate(cx[n])
            icx[n] = cumsimps(cx[n])

        gamma_model = lf.Model(exact, 't')

        try:
            results
        except:
            results = {}

        gamma_params = gamma_model.make_params()

        # in general, 1.1-20
        gamma_params.add('b',  min=1.1,   max=20, value=2)
        gamma_params.add('mrt', min=0.5,    max=10,
                         value=1)     # 1e-15 - 1e4

        uncs = np.std(icx, axis=0)
        last = 1000  # L#2000
        first = 10

        feedback = True

        for n in range(N):
            print(n)
        #     starttime = time.time()
            this_integral = icx[n][first:last]
            plt.figure()
            plt.plot(this_integral)
            plt.plot(uncs[first:last])
            plt.yscale('log')
            plt.savefig('asdf.png')
            results[n] = \
                gamma_model.fit(this_integral, gamma_params, t=np.arange(first, last),
                                method='basinhopping', weights=uncs[first:last]**-2,
                                #                     fit_kws={'nwalkers': nwalkers, 'steps': nsteps, #10000
                                #                         'pos': initials, 'workers': 16},
                                calc_covar=True
                                )
        if feedback:
            plt.figure()
            for n in range(N):
                # display(results[n])
                plt.plot(
                    t, exact(t, *list(results[n].best_values.values())), 'C0', alpha=0.5)
                plt.plot(np.arange(first, last),
                         icx[n][first-1:last-1], 'C1', alpha=0.5)
                plt.xscale('log')
            #     plt.yscale('log')
            #     plt.ylim(bottom = 1)
                plt.ylim(bottom=0)
            plt.plot(exact(t, b, thetime), 'k')
            plt.savefig(f'{b}-{10**thetime:0.1f}-feedback.png')

        mdcx = np.diff(np.mean(cx, axis=0))
        mdcx[0]
        for j in range(L):
            if mdcx[j] > 0:
                platstart = j
                print(j, mdcx[j])
                break

        temp = icx[:, int(platstart*0.9):platstart*2]
        plt.figure()
        thetimes = np.zeros(N)
        plats = np.zeros(N)
        for n in range(N):
            thetimes[n] = 10**results[n].best_values['mrt']
            plt.scatter(n, thetimes[n], c='C0')
            plats[n] = np.mean(temp[n])
            plt.scatter(n, plats[n], c='C1')

        m0 = np.mean(np.log10(thetimes))
        s0 = np.std(np.log10(thetimes))/np.sqrt(N-1)

        mm, ss = lognormal_normal([m0, s0])
        print(mm, ss)

        plt.axhline(mm, color='C0')
        plt.axhline(mm+ss, color='C0')
        plt.axhline(mm-ss, color='C0')
        estimates[b-1, i, 0, :] = mm, ss

        mm = np.mean(plats)
        ss = np.std(plats)/np.sqrt(N-1)
        estimates[b-1, i, 1, :] = mm, ss
        plt.axhline(mm, color='C1')
        plt.axhline(mm+ss, color='C1')
        plt.axhline(mm-ss, color='C1')
        plt.axhline(10**thetime, color='k')
        plt.ylim(bottom=0)
        plt.savefig(f'{b}-{10**thetime:0.1f}-stats2.png')
with open('estimates2.txt', 'wb') as f:
    np.save(f, estimates)
