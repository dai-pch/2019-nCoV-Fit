from scipy.integrate import odeint
from scipy.optimize import minimize
from itertools import product, repeat
from functools import reduce
import math as mt
from random import random
import numpy as np
from multiprocessing import Pool

class InfectModel(object):
    def __init__(self, paras):
        self.paras = paras

    def __call__(self, vars, t):
        return self.step(vars, t)

    def step(self, vars, t):
        pass

    def get_i(self, vars):
        pass

    def get_r(self, vars):
        pass


class SIR(InfectModel):
    def __init__(self, paras):
        self.paras = tuple(map(abs, paras))
        
    def step(self, vars, t):
        # 给出变量矢量sir，和两个参数a, b计算出
        # ds/dt, di/dt, dr/dt的值
        a, b = self.paras
        s, i, r = vars
        N = s + i + r
        return np.array([-a*s*i / N, a*s*i / N - b*i, b*i])

    def get_i(self, vars):
        return vars[1]

    def get_r(self, vars):
        return vars[2]


# def SIR(sir, t, a, b):
#     # 给出变量矢量sir，和两个参数a, b计算出
#     # ds/dt, di/dt, dr/dt的值
#     s, i, r = sir
#     N = s + i + r
#     return np.array([-a*s*i / N, a*s*i / N - b*i, b*i])

class SEIR(InfectModel):
    def __init__(self, paras):
        self.paras = tuple(map(abs, paras))

    def step(self, vars, t):
        # beta S->E
        # sigma E->I
        # gamma I->R
        beta, sigma, gamma = self.paras
        s, e, i, r = tuple(map(abs, vars))
        N = s + e + i + r
        return np.array([-beta*s*i / N, 
                        beta*s*i / N - sigma*e, 
                        sigma*e - gamma*i, 
                        gamma*i])

    def get_i(self, vars):
        return vars[2]

    def get_r(self, vars):
        return vars[3]


# beta S->E
# sigma E->I
# gamma I->R
# def SEIR(seir, t, beta, gamma, sigma):
#     s, e, i, r = seir
#     N = s + e + i + r
#     return np.array([-beta*s*i / N, 
#                     beta*s*i / N - sigma*e, 
#                     sigma*e - gamma*i, 
#                     gamma*i])

# calculate sir model in some points with certain parameters and initial values
# paras: a, b / beta, sigma, gamma
# inits: s, i, r / s, e, i, r
def solve(model, inits, t_max, points=5001):
    inits = tuple(map(abs, inits))
    step = t_max / (points - 1)
    ts = np.arange(0, t_max + 2*step, step) # 创建时间点，单位为天
    solve = odeint(model, inits, ts)
    return solve, ts

# error function
# paras: a, b, s0, i0, r0
# s0 is the initial value of s.
# golden_ts: [t0, t1, ..., tn]
# golden_vs: [[i, r], ..., []]
def err(model, inits, golden_ts, golden_vs):
    inits = tuple(map(abs, inits))
    points = 801
    t_max = max(golden_ts)
    sv, ts = solve(model, inits, t_max, points)
    step = ts[1]
    outs = []
    for tt in golden_ts:
        t_id_l = mt.floor(tt / step)
        l_intev = tt / step - t_id_l 
        t_id_r = t_id_l + 1
        out = sv[t_id_l] * (1 - l_intev) + sv[t_id_r] * l_intev
        outs.append(out)
    n = len(golden_ts)
    err = 0
    for ii in range(n):
        ei = model.get_i(outs[ii]) - golden_vs[ii][0]
        er = model.get_r(outs[ii]) - golden_vs[ii][1]
        e = ei * ei + er * er
        err += e
    return err / (2*n)

# find local minimum
def find_local_minimum(array, proc):
    sz = array.shape
    with Pool(processes=proc) as pool:
        cords = product(*tuple(map(lambda x: range(0, x - 1), sz)))
        res = pool.starmap(is_local_minimum, zip(repeat(array), cords))
    res = [e for e in res if e]
    return res

def is_local_minimum(array, cord):
    sz = array.shape
    for dim_id, xi in enumerate(cord):
        if sz[dim_id] <= 2:
            continue
        cord_l = cord[0:dim_id] + tuple([xi - 1]) + cord[dim_id+1:]
        cord_r = cord[0:dim_id] + tuple([xi + 1]) + cord[dim_id+1:]
        if ((cord[dim_id] != 0 and array[cord] > array[cord_l]) 
            or (array[cord] > array[cord_r])):
            return []
    return cord

# optimization
# para_ranges: [(para0_min, para0_max, err), 
#               (para1_min, para1_max, err), 
#               ..., 
#               (paran_min, paran_max, err)] 
# golden_ts, golden_vs is same to function sir_err
def grid_search(target_func, para_ranges, n=10, proc=16):
    seg_per_para = 16
    dim = len(para_ranges)
    para_lengths = [(para_ranges[i][1] - para_ranges[i][0]) for i in range(dim)]
    para_lefts = [para_ranges[i][0] for i in range(dim)]
    para_errs = [para_ranges[i][2] for i in range(dim)]
    para_segs = tuple(map(lambda id: 1 if para_lengths[id] < para_errs[id] else seg_per_para, range(dim)))
    para_steps = tuple(map(lambda id: para_lengths[id] / para_segs[id], range(dim)))
    para_cnts = tuple(map(lambda x: x + 1, para_segs))
    err_grid = np.zeros(para_cnts)
    # err_grid_2 = np.zeros(para_cnts)
    if proc > 1:
        err_grid_res = []
        with Pool(processes=proc) as pool:
            for grid in product(*(map(range, para_cnts))):
                paras = [para_lefts[ii] + grid[ii] * para_steps[ii] for ii in range(dim)]
                err_grid_res.append((grid, pool.apply_async(target_func, paras)))
            for res in err_grid_res:
                err_grid[res[0]] = res[1].get()
    else:
    # if True:
        for grid in product(*(map(range, para_cnts))):
            paras = [para_lefts[ii] + grid[ii] * para_steps[ii] for ii in range(dim)]
            err_grid[grid] = target_func(*paras)
    # check
    # if (err_grid_2 == err_grid).all():
    #     print("correct")
    # else:
    #     print(err_grid_2)
    #     print(err_grid)
    #     exit()
    # find extremum
    extremums = []
    extremum_errs = []
    # find global minimum
    min_err = err_grid.min()
    # print(min_err)
    min_idxs = np.array(np.where(err_grid == min_err)).transpose()
    min_idxs = [tuple(min_idx) for min_idx in min_idxs]
    extremums += min_idxs
    extremum_errs += [err_grid[e] for e in min_idxs]
    # find other extremum
    t = find_local_minimum(err_grid, proc)
    # for grid in product(*tuple(map(lambda x: range(0, x - 1), para_cnts))):
    #     flag = True
    #     for dim_id, xi in enumerate(grid):
    #         if para_cnts[dim_id] <= 2:
    #             continue
    #         neibour_grid_l = grid[0:dim_id] + tuple([xi - 1]) + grid[dim_id+1:]
    #         # neibour_grid_l -= 1
    #         neibour_grid_r = grid[0:dim_id] + tuple([xi + 1]) + grid[dim_id+1:]
    #         # neibour_grid_r = grid
    #         # neibour_grid_r[dim_id] += 1
    #         if ((grid[dim_id] != 0 and err_grid[grid] > err_grid[neibour_grid_l]) 
    #             or (err_grid[grid] > err_grid[neibour_grid_r])):
    #             flag = False
    #             break
    #     if flag and not grid in extremums:
    #         extremums.append(grid)
    #         err = err_grid[grid]
    #         extremum_errs.append(err)
    for grid in t:
        if not grid in extremums:
            extremums.append(grid)
            err = err_grid[grid]
            extremum_errs.append(err)
    # prune
    extremum_zip = list(zip(extremum_errs, extremums))
    extremum_zip = sorted(extremum_zip, key=lambda x: x[0])
    extremums = [extremum_zip[i][1] for i in range(min(n, len(extremum_zip)))]
    # does all err is satisfied
    satisfied = reduce(lambda x, y: x and y, map(lambda ii: para_lengths[ii] < para_errs[ii], range(dim)))    
    if satisfied:
        res = []
        for extremum in extremums:
            t = tuple(map(
                    lambda dim_id: para_lefts[dim_id] + para_steps[dim_id] * extremum[dim_id], 
                    range(dim)))
            res.append((t, err_grid[extremum]))
        return res
    # fine search
    new_ranges = []
    for extremum in extremums:
        new_range = list(map(lambda dim_id: (para_lefts[dim_id] + para_steps[dim_id] * 
                                            (0 if extremum[dim_id] == 0 
                                            else extremum[dim_id] - 1), 
                                        para_lefts[dim_id] + para_steps[dim_id] * 
                                            (para_cnts[dim_id] - 1 if extremum[dim_id] == para_cnts[dim_id] - 1
                                            else extremum[dim_id] + 1), 
                                        para_errs[dim_id]),
                        range(dim)))
        new_ranges.append(new_range)
    # print(new_ranges)
    # exit()
    res = []
    for new_range in new_ranges:
        res += grid_search(target_func, new_range)
    return res

def test_search():
    func = lambda x, y, z: mt.sin(x*2*mt.pi) + mt.sin(2*mt.pi*y) + mt.cos(2*mt.pi*z)
    mins = grid_search(func, [(-1.1, 1, 0.01), (-1, 1.1, 0.01), (-1, 1.1, 0.01)], 10)   
    print(mins)

def find_n_res(search_res, n):
    sorted(search_res, key=lambda x: x[1])
    return search_res[0:n]

# data
datas = [
    (18+23, [122, 3]),
    (19+23, [199, 4]),
    (20+23, [291, 5]),
    (21+23, [440, 37]),
    (22+23, [686, 45]),
    (22+15/24+23, [616+395, 28+17]),
    (22+23/24+23, [639+422, 30+17]),
    (24+12.5/24+23, [880+1072, 34+26]),
    (25+23, [897+1076, 36+26]),
    (25+10.5/24+23, [1303+1965, 38+41])
]
golden_ts = [datas[i][0]-18-23 for i in range(len(datas))]
golden_vs = [datas[i][1] for i in range(len(datas))]


def fit(func, rg, n):
    mins = grid_search(func, rg)
    mins = find_n_res(mins, n)
    # print(mins)
    return mins

# sir fit
def sir_fit_func(a, b, s0, i0, r0): 
    return err(SIR((a, b)), (s0, i0, r0), golden_ts, golden_vs)

def sir_fit(rg, n):
    return fit(sir_fit_func, rg, n)
    
# seir fit
def seir_fit_func(beta, sigma, gamma, s0, e0, i0): 
    return err(SEIR((beta, sigma, gamma)), 
                    (s0, e0, i0, 0), golden_ts, golden_vs)
def seir_fit(n):
    # beta, sigma, gamma, s0, e0, i0
    rgs = [(0.001, 1, 0.0001), 
            (0.001, 0.5, 0.0001), 
            (0.001, 0.5, 0.0001), 
            (1000, 10000000, 10), 
            (0, 4000, 1), (0, 10, 1)]
    return fit(seir_fit_func, rgs, n)

def seir_fit2_once(dummy):
    rgs = [(0.001, 1, 0.0001), 
            (0.001, 0.5, 0.0001), 
            (0.001, 0.5, 0.0001), 
            (1000, 1000000, 10), 
            (0, 4000, 1), (0, 10, 1)]
    bnds = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None))
    def func(p): 
        beta, sigma, gamma, s0, e0, i0 = p
        return err(SEIR((beta, sigma, gamma)), 
                (s0, e0, i0, 0), golden_ts, golden_vs)
    init = []
    for rg in rgs:
        e = random() * (rg[1] - rg[0]) + rg[0]
        init.append(e)
    r = minimize(func, init, method='nelder-mead')#,bounds=bnds)
    return (r.x, r.fun)

def seir_fit2(n, times=100, batch = 10, proc = 10):
    rgs = [(0.001, 5, 0.000), 
            (0.001, 0.5, 0.0001), 
            (0.001, 0.5, 0.0001), 
            (1000, 100000, 10), 
            (1, 5000, 1), (0, 10, 1)]
    init = np.array([0.4, 0.3, 0.016, 
                    140000, 100, 1])
    res = []
    t = 1
    batch = max(batch, n)
    with Pool(processes=proc) as pool:
        while t < times:
            print("Run ", t)
            t += batch
            res += pool.map(seir_fit2_once, range(batch))
            res = sorted(res, key=lambda x: x[1])
            res = [res[i] for i in range(n)]
            print(n, "-th error ", res[-1][1])
            if res[-1][1] < 3000:
                return res
    return res

# 调用ode对lorenz进行求解, 用两个不同的初始值
# track1 = odeint(lorenz, (0.0, 1.00, 0.0), t, args=(10.0, 28.0, 3.0))
# track2 = odeint(lorenz, (0.0, 1.01, 0.0), t, args=(10.0, 28.0, 3.0))

# 绘图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

def sir_draw(res, golden_ts, golden_vs):
    ((a, b, s0, i0, r0), err) = res
    sv, ts = solve(SIR((a, b)), (s0, i0, r0), 180, 10000)
    # fig = plt.figure()
    # ax = fig.add_axes()
    fig, ax = plt.subplots()
    ax.plot(ts, sv[:, 0])
    ax.plot(ts, sv[:, 1])
    ax.plot(ts, sv[:, 2])
    ax.legend(["易感", "感染", "恢复/死亡"])
    # ax.plot(track2[:,0], track2[:,1], track2[:,2])
    # data
    v = [vs[0] for vs in golden_vs]
    ax.scatter(golden_ts, v, marker='.')
    tt = "err=%.2f;a=%.6f;b=%.6f;S0=%.2f;i0=%.2f" % (err, a, b, s0, i0)
    plt.title(tt)
    # plt.show()
    plt.savefig('sir-'+tt+'.png')
    plt.clf()


def seir_draw(res, golden_ts, golden_vs):
    ((beta, sigma, gamma, s0, e0, i0), err) = res
    sv, ts = solve(SEIR((beta, sigma, gamma)), (s0, e0, i0, 0), 180, 10000)
    # fig = plt.figure()
    # ax = fig.add_axes()
    fig, ax = plt.subplots()
    ax.plot(ts, sv[:, 0])
    ax.plot(ts, sv[:, 1])
    ax.plot(ts, sv[:, 2])
    ax.plot(ts, sv[:, 3])
    ax.legend(["易感", "潜伏", "感染", "恢复/死亡"])
    # ax.plot(track2[:,0], track2[:,1], track2[:,2])
    # data
    v = [vs[0] for vs in golden_vs]
    ax.scatter(golden_ts, v, marker='.')
    tt = "err=%.2f;b=%.6f;s=%.6f;g=%.6f;S0=%.2f;E0=%.2f;I0=%.2f" % (err, beta, sigma, gamma, s0, e0, i0)
    plt.title(tt)
    # plt.show()
    plt.savefig('seir-'+tt+'.png')
    plt.clf()


def sir_run():
    # a, b, s0, i0
    rg = [(0.0000001, 10, 0.0001), (0.0000001, 0.5, 0.0001), (1000, 1000000, 10), (1, 10000, 0.5), (3, 100, 0.5)]
    res = sir_fit(rg, 80)
    print(res)
    for r in res:
        sir_draw(r, golden_ts, golden_vs)


def seir_run():
    res = seir_fit(80)
    print(res)
    for r in res:
        seir_draw(r, golden_ts, golden_vs)


def seir_run2():
    res = seir_fit2(10, batch=64, times=2000, proc=16)
    print(res)
    for r in res:
        seir_draw(r, golden_ts, golden_vs)

def main():
    # test_search()
    # res = [((0.00012969970703125, 0.0, 822.0977783203125), 2076.230989456712)]
    sir_run()
    # seir_run2()
    # seir_run()


if __name__ == "__main__":
    main()
