import gurobipy as gp
import numpy
import csv
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

 
with open('src/CovMatrixNorm.csv', 'r') as f:
    reader = csv.reader(f,delimiter=',')
    temp1 = list(reader)

W = [[float(j) for j in i] for i in temp1]

NumAsset=20

N = range(NumAsset)

# Small adjustment to make positive definite
for i in N:
    W[i][i] += 0.005

R=[
1.071877684,
1.075411639,
1.10823578,
1.07868473,
1.103370352,
1.312741894,
1.117739229,
1.058670271,
1.178226399,
1.063677793,
1.133877788,
1.128557584,
1.010831389,
1.096413424,
1.120902431,
1.01264185,
1.061801832,
1.131959114,
1.119030185,
1.115390706
]

# Add the risk free asset
RiskFree=True
if RiskFree:
    R.append(1.02)
    for i in N:
        W[i].append(0.0)
    N = range(NumAsset+1)
    W.append([0.0 for i in N])


Names=[
'AMC',
'ANZ',
'BHP',
'BXB',
'CBA',
'CSL',
'GMG',
'IAG',
'MQG',
'NAB',
'NCM',
'RIO',
'SCG',
'SUN',
'TCL',
'TLS',
'WBC',
'WES',
'WOW',
'WPL',
'CSH'
]

Version = 2

if Version==1:
    m = gp.Model('Markowitz')
    X = {i: m.addVar() for i in N}
    Y = {i: m.addVar(vtype=gp.GRB.BINARY) for i in N}
    m.addConstr(gp.quicksum(X[i] for i in N)==1)
    YUpperBound = {i: m.addConstr(X[i]<=Y[i]) for i in N}
    YLowerBound = {i: m.addConstr(X[i]>=0.1*Y[i]) for i in N}
    m.setParam('OutputFlag',0)    
    Ret = []
    SD = []
    
    for l in range(1,100):
        Lambda = 0.01*l    
        m.setObjective(Lambda*gp.quicksum(R[i]*X[i] for i in N)-
               (1-Lambda)*gp.quicksum(W[i][j]*X[i]*X[j] for i in N for j in N),
               gp.GRB.MAXIMIZE)    
        m.optimize()
        Ret.append(sum(R[i]*X[i].x for i in N))
        SD.append(math.sqrt(sum(W[i][j]*X[i].x*X[j].x for i in N for j in N)))
    
    plt.plot(SD,Ret)
    plt.show()
    
if Version==2:
    m = gp.Model('SAA CVar')
    alpha = 0.05
    S = range(10000)
    numpy.random.seed(95)
    RS = numpy.random.multivariate_normal(R, W, len(S)).tolist()
    X = {i: m.addVar() for i in N}
    m.addConstr(gp.quicksum(X[i] for i in N)==1)
    Beta = {s: m.addVar() for s in S}
    Betam = {s: m.addVar() for s in S}
    Var = m.addVar()
    CVar = m.addVar()
    SetBeta = {s: 
        m.addConstr(Beta[s]==gp.quicksum(RS[s][i]*X[i] for i in N)) 
        for s in S}
    SetBetam = {s: m.addConstr(Beta[s]+Betam[s]>=Var) for s in S}
    m.addConstr(CVar==Var-gp.quicksum(Betam[s] for s in S)/(alpha*len(S)))

    m.setParam('OutputFlag',0)    
    Ret = []
    CV = []
    LV = []
    
    l = 0.5
    m.setObjective(l*gp.quicksum(Beta[s] for s in S)/len(S)+
               (1-l)*CVar,
               gp.GRB.MAXIMIZE) 
    m.optimize()
    print('1', sum([Beta[s].X for s in S])/len(S))
    print('2', CVar.X)
    
    # for l in range(1,100):
    #     Lambda = 0.01*l    
    #     m.setObjective(Lambda*gp.quicksum(Beta[s] for s in S)/len(S)+
    #            (1-Lambda)*CVar,
    #            gp.GRB.MAXIMIZE)    
    #     m.optimize()
    #     Ret.append(sum(Beta[s].x for s in S)/len(S))
    #     CV.append(CVar.x)
    #     LV.append(Lambda)
    
    # plt.plot(LV,Ret)
    # plt.plot(LV,CV)
    # plt.show()
    
if Version==3:
    m = gp.Model('Normal Chance')
    LowReturn = 0.9
    Confidence = 0.95
    FInv = norm.ppf(Confidence)
    X = {i: m.addVar() for i in N}
    m.addConstr(gp.quicksum(X[i] for i in N)==1)
    m.setObjective(gp.quicksum(R[i]*X[i] for i in N), gp.GRB.MAXIMIZE)
    Z = m.addVar()
    m.addConstr(Z==gp.quicksum(R[i]*X[i] for i in N)-LowReturn)    
    m.addConstr(FInv**2*gp.quicksum(W[i][j]*X[i]*X[j] for i in N for j in N)
                <=Z*Z)
    m.optimize()
    S = range(100000)
    numpy.random.seed(95)
    RS = numpy.random.multivariate_normal(R, W, len(S)).tolist()
    XV = [X[i].x for i in N]
    Fail = [s for s in S if sum(RS[s][i]*XV[i] for i in N)<LowReturn]
    print('Failed:', len(Fail)/len(S))
    
if Version==4:
    m = gp.Model('SAA Chance')
    LowReturn = 0.9
    Confidence = 0.95
    S = range(2500)
    k = 24 #round((1-Confidence)*len(S))
    numpy.random.seed(95)
    # Square the returns so they are no longer Normal
    RS = numpy.random.multivariate_normal(R, W, len(S))**2
    RS = RS.tolist()

    X = {i: m.addVar() for i in N}
    Y = {s: m.addVar(vtype=gp.GRB.BINARY) for s in S}
    m.addConstr(gp.quicksum(X[i] for i in N)==1)
    m.addConstr(gp.quicksum(Y[s] for s in S)<=k)
    ScenarioConstraint = {s:
        m.addConstr(gp.quicksum(RS[s][i]*X[i] for i in N)>=LowReturn*(1-Y[s]))
        for s in S}
    m.setObjective(gp.quicksum(R[i]*X[i] for i in N), gp.GRB.MAXIMIZE)
    m.optimize()
    
    S = range(100000)
    numpy.random.seed(91)
    RS = numpy.random.multivariate_normal(R, W, len(S))**2
    RS = RS.tolist()
    XV = [X[i].x for i in N]
    Fail = [s for s in S if sum(RS[s][i]*XV[i] for i in N)<LowReturn]
    print('Failed:', len(Fail)/len(S))
    
if Version==5:   
    m = gp.Model('SAA Chance')
    LowReturn = 0.9
    Confidence = 0.95
    S = range(100000)
    k = 3922 #round((1-Confidence)*len(S))
    numpy.random.seed(95)
    # Square the returns so they are no longer Normal
    RS = numpy.random.multivariate_normal(R, W, len(S))**2
    print('Sample done')
    RS = RS.tolist()
    cov = numpy.cov(RS,rowvar=False)

    Buffer = 0.9*norm.ppf(Confidence)
    X = {i: m.addVar() for i in N}
    m.addConstr(gp.quicksum(X[i] for i in N)==1)
    m.setObjective(gp.quicksum(R[i]*X[i] for i in N), gp.GRB.MAXIMIZE)
    Z = m.addVar()
    m.addConstr(Z==gp.quicksum(R[i]*X[i] for i in N)-LowReturn)    
    m.setParam('OutputFlag',0)
    def SolveBuffer(b):            
        tCon = m.addConstr(
                    b**2*gp.quicksum(cov[i][j]*X[i]*X[j] for i in N for j in N)
                    <=Z*Z)
        m.optimize()
        XV = [X[i].x for i in N]
        Fail = [s for s in S if sum(RS[s][i]*XV[i] for i in N)<LowReturn]       
        m.remove(tCon)
        return m.objVal, len(Fail)
    # Assume we know the starting conditions
    LowFail = SolveBuffer(Buffer)[1]
    HighFail = SolveBuffer(Buffer/2)[1]
    LowBuffer = Buffer
    HighBuffer = Buffer/2
    while HighFail-LowFail>1 and LowFail!=k:
        NewBuffer = HighBuffer+(LowBuffer-HighBuffer)*(k-LowFail)/(HighFail-LowFail)
        r = SolveBuffer(NewBuffer)
        print(NewBuffer,r)
        if r[1] > k:
            HighFail = r[1]
            HighBuffer = NewBuffer
        else:
            LowFail = r[1]
            LowBuffer = NewBuffer
    print('Ratio', k/len(S))
    S = range(100000)
    numpy.random.seed(91)
    RS = numpy.random.multivariate_normal(R, W, len(S))**2
    RS = RS.tolist()
    XV = [X[i].x for i in N]
    Fail = [s for s in S if sum(RS[s][i]*XV[i] for i in N)<LowReturn]
    print('Failed:', len(Fail)/len(S))
    
