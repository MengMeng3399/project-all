import taichi as ti
@ti.data_oriented
class CGSolver:
    #输入的参数依次为，矩阵大小m n
    def __init__(self, m, n, u, v, cell_type):
        self.m = m
        self.n = n
        self.u = u
        self.v = v
        self.cell_type = cell_type
        # 右侧的线性系统：
        self.b = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        # 左侧的线性系统
        self.Adiag = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.Ax = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.Ay = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        # cg需要的参数
        #p:x
        self.p = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        #r:残差
        self.r = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        #s: d
        self.s = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        #As:Ad
        self.As = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        #sum:rTr
        self.sum = ti.field(dtype=ti.f32, shape=())
        #alpha:往一个方向走的距离大小
        self.alpha = ti.field(dtype=ti.f32, shape=())
        #beta：
        self.beta = ti.field(dtype=ti.f32, shape=())


    @ti.kernel
    def system_init_kernel(self, scale_A: ti.f32, scale_b: ti.f32):
        # 右边线性系统
        for i, j in ti.ndrange(self.m, self.n):
            if self.cell_type[i, j] ==1:
                self.b[i,j] = -1 * scale_b * (self.u[i + 1, j] - self.u[i, j] +
                                              self.v[i, j + 1] - self.v[i, j])

        for i, j in ti.ndrange(self.m, self.n):
            if self.cell_type[i, j] == 1 and i-1>=0 and j-1>=0 and i+1<=self.m-1 and j+1<=self.n-1:
                if self.cell_type[i - 1, j] == -2:
                    self.b[i, j] -= scale_b * (self.u[i, j] - 0)
                if self.cell_type[i + 1, j] == -2:
                    self.b[i, j] += scale_b * (self.u[i + 1, j] - 0)
                if self.cell_type[i, j - 1] == -2:
                    self.b[i, j] -= scale_b * (self.v[i, j] - 0)
                if self.cell_type[i, j + 1] == -2:
                    self.b[i, j] += scale_b * (self.v[i, j + 1] - 0)

        #左侧线性系统：
        for i, j in ti.ndrange(self.m, self.n):
            #因为对称，在这里只关心 右 ，上 方向
            if self.cell_type[i, j] ==1 and i-1>=0 and j-1>=0 and i+1<=self.m-1 and j+1<=self.n-1 :
                if self.cell_type[i - 1, j] == 1:
                    self.Adiag[i, j] += scale_A
                if self.cell_type[i + 1, j] == 1:
                    self.Adiag[i, j] += scale_A
                    self.Ax[i, j] = -scale_A
                elif self.cell_type[i + 1, j] == -1:
                    self.Adiag[i, j] += scale_A
                if self.cell_type[i, j - 1] == 1:
                    self.Adiag[i, j] += scale_A
                if self.cell_type[i, j + 1] == 1:
                    self.Adiag[i, j] += scale_A
                    self.Ay[i, j] = -scale_A
                elif self.cell_type[i, j + 1] == -1:
                    self.Adiag[i, j] += scale_A


    def system_init(self, scale_A, scale_b):
        self.b.fill(0.0)
        self.Adiag.fill(0.0)
        self.Ax.fill(0.0)
        self.Ay.fill(0.0)
        self.system_init_kernel(scale_A, scale_b)
        #
    def solve(self, max_iters):

        tol =1e-6
        self.p.fill(0.0)
        self.As.fill(0.0)
        self.s.fill(0.0)

        #该系统从原点出发
        self.r.copy_from(self.b)
        self.reduce(self.r, self.r)
        init_rTr = self.sum[None]

        # print("init rTr = {}".format(init_rTr))
        if init_rTr < tol:

            print("Converged: init rtr = {}".format(init_rTr))
            # print("zhixing")

        else:

            self.s.copy_from(self.r)
            old_rTr = init_rTr

            for i in range(max_iters):
                # alpha = rTr / sAs
                #As=A*d
                self.compute_As()
                #dTq
                self.reduce(self.s, self.As)
                sAs = self.sum[None]
                if sAs==0:
                    break
                self.alpha[None] = old_rTr / sAs
                # p = p + alpha * s
                self.update_p()
                # r = r - alpha * As
                self.update_r()

                # 检查收敛性
                self.reduce(self.r, self.r)
                rTr = self.sum[None]
                if rTr < init_rTr * tol:
                    break
                new_rTr = rTr
                self.beta[None] = new_rTr / old_rTr
                # s = r + beta * s
                self.update_s()
                old_rTr = new_rTr
                i+=1

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = 0.0
        for i, j in ti.ndrange(self.m, self.n):
            if self.cell_type[i, j] == 1:
                self.sum[None] += p[i, j] * q[i, j]

    @ti.kernel
    def compute_As(self):
        for i, j in ti.ndrange(self.m, self.n):
            if self.cell_type[i, j] == 1 and i-1>=0 and j-1>=0 and i+1<=self.m-1 and j+1<=self.n-1:
                self.As[i, j] = self.Adiag[i, j] * self.s[i, j] + self.Ax[
                    i - 1, j] * self.s[i - 1, j] + self.Ax[i, j] * self.s[
                                    i + 1, j] + self.Ay[i, j - 1] * self.s[
                                    i, j - 1] + self.Ay[i, j] * self.s[i, j + 1]
    @ti.kernel
    def update_p(self):
        for i, j in ti.ndrange(self.m, self.n):
            if self.cell_type[i, j] ==1:
                self.p[i, j] = self.p[i, j] + self.alpha[None] * self.s[i, j]

    @ti.kernel
    def update_r(self):
        for i, j in ti.ndrange(self.m, self.n):
            if self.cell_type[i, j] == 1:
                self.r[i, j] = self.r[i, j] - self.alpha[None] * self.As[i, j]

    @ti.kernel
    def update_s(self):
        for i, j in ti.ndrange(self.m, self.n):
            if self.cell_type[i, j] == 1:
                self.s[i, j] = self.r[i, j] + self.beta[None] * self.s[i, j]
