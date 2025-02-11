
# evs = [13.781534526986075,15.700066364271223,15.737047727094673,15.790342074839193,15.821452455587186,15.855991604539645,15.904877310960563,15.933075156547586,15.948302865168547,15.959308488615005,15.96498391956805]
# cvars = [11.287111975486702,11.285633615901943,11.278952661489166,11.261125383318562,11.244588453113735,11.215981331362125,11.156136206727323,11.103285633278698,11.058166815937497,10.99483376977423,0.0]

# evs = [15.052850257345888,16.47297082139597,16.54146044984803,16.60087314527109,16.653533459294263,16.706880821620935,16.741705943502897,16.773063876854582,16.787440291102126,16.793594091985334,16.799386922494286]
# cvars = [12.60533146286062,12.601915679100461,12.58964475628449,12.569984713158986,12.541692398146424,12.498785999771933,12.455990293764255,12.39722129835977,12.35489045538274,12.31801202632933,0.0]

# import matplotlib.pyplot as plt

# xvalues = [Lambda for Lambda in range(10, 91, 10)]

# plt.plot(xvalues, evs[1:-1], label = 'Expected Value')
# plt.plot(xvalues, cvars[1:-1], label = 'CVaR')
# plt.title('Expected Value and CVaR Over Lambda')
# plt.legend()
# plt.show()

props = [
0.8293652156236265,
0.8580727490523397,
0.9080917264688091,
0.9257305990599591,
0.9737671662686429,
0.9821306736907093,
0.99176476217142,
]

import matplotlib.pyplot as plt

plt.bar(range(1, 8), props)
plt.ylim(0.8, 1.0)
plt.ylabel('Average Profit Proportion to Eight-Day Window')
plt.xlabel('Length of Window')
plt.title('Effect of Window Length on Profit in Simulations\nOver 50 Samples')
plt.show()