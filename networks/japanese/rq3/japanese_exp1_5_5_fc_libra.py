
#
assume(0 <= x04 <= 0)
assume(1 <= x05 <= 1)
assume(0 <= x06 <= 0)
assume(1 <= x07 <= 1)
assume(0 <= x08 <= 0)
assume(0 <= x09 <= 0)
assume(x15 <= 0.5)
assume(1 <= x18 <= 1)
assume(0 <= x19 <= 0)
assume(0 <= x20 <= 0)
assume(x21 <= 0.5)
assume(x22 <= 0.5)


x10 = (-0.185425)*x00 + (-0.135919)*x01 + (-0.143197)*x02 + (-0.135521)*x03 + (-0.097338)*x04 + (0.199838)*x05 + (-0.056929)*x06 + (0.023663)*x07 + (-0.212125)*x08 + (0.185393)*x09 + (-0.050074)*x010 + (-0.191363)*x011 + (0.098667)*x012 + (-0.056381)*x013 + (0.144326)*x014 + (-0.146976)*x015 + (-0.205532)*x016 + (-0.189742)*x017 + (0.045291)*x018 + (-0.023384)*x019 + (-0.193172)*x020 + (0.034044)*x021 + (-0.014797)*x022 + (0.003572)
x11 = (0.075330)*x00 + (-0.106067)*x01 + (-0.148856)*x02 + (0.142120)*x03 + (-0.100337)*x04 + (0.146676)*x05 + (0.169065)*x06 + (-0.145821)*x07 + (-0.096134)*x08 + (0.025101)*x09 + (-0.196601)*x010 + (0.064036)*x011 + (-0.086880)*x012 + (-0.092451)*x013 + (-0.161196)*x014 + (0.097547)*x015 + (-0.132875)*x016 + (-0.166666)*x017 + (-0.054834)*x018 + (-0.116080)*x019 + (-0.202717)*x020 + (0.084034)*x021 + (-0.079333)*x022 + (-0.033520)
x12 = (0.157361)*x00 + (0.231615)*x01 + (-0.044845)*x02 + (-0.048645)*x03 + (-0.069998)*x04 + (-0.019196)*x05 + (0.138002)*x06 + (0.070689)*x07 + (-0.007530)*x08 + (0.197744)*x09 + (0.387281)*x010 + (-0.941566)*x011 + (0.851391)*x012 + (-0.345491)*x013 + (0.553548)*x014 + (0.051104)*x015 + (0.000022)*x016 + (-0.143006)*x017 + (-0.042381)*x018 + (0.082711)*x019 + (-0.016584)*x020 + (-0.235160)*x021 + (0.250992)*x022 + (0.092287)
x13 = (0.206064)*x00 + (0.082847)*x01 + (-0.104897)*x02 + (0.157506)*x03 + (0.138437)*x04 + (0.134392)*x05 + (-0.055732)*x06 + (0.038852)*x07 + (-0.003816)*x08 + (0.196307)*x09 + (0.248004)*x010 + (0.119063)*x011 + (-0.173774)*x012 + (0.195496)*x013 + (0.096806)*x014 + (0.101342)*x015 + (-0.194294)*x016 + (0.272156)*x017 + (0.011794)*x018 + (-0.182548)*x019 + (-0.130793)*x020 + (0.143436)*x021 + (0.059262)*x022 + (-0.048231)
x14 = (-0.353739)*x00 + (0.100959)*x01 + (0.156487)*x02 + (0.118950)*x03 + (0.120769)*x04 + (-0.096600)*x05 + (0.049136)*x06 + (0.072660)*x07 + (0.047409)*x08 + (-0.149974)*x09 + (0.028731)*x010 + (-0.198664)*x011 + (0.412772)*x012 + (-0.192208)*x013 + (0.118037)*x014 + (0.018206)*x015 + (-0.264352)*x016 + (-0.142817)*x017 + (0.001836)*x018 + (-0.168818)*x019 + (-0.085491)*x020 + (-0.029604)*x021 + (-0.077234)*x022 + (-0.181095)
#
ReLU(x10)
ReLU(x11)
ReLU(x12)
ReLU(x13)
ReLU(x14)

x20 = (-0.405311)*x10 + (0.277448)*x11 + (0.825650)*x12 + (-0.262189)*x13 + (0.168815)*x14 + (-0.138922)
x21 = (-0.230358)*x10 + (-0.286076)*x11 + (-0.226602)*x12 + (-0.213065)*x13 + (0.061624)*x14 + (-0.050787)
x22 = (-0.051263)*x10 + (0.031357)*x11 + (-0.342714)*x12 + (-0.280967)*x13 + (-0.387162)*x14 + (-0.269968)
x23 = (-0.316831)*x10 + (0.374504)*x11 + (0.602639)*x12 + (-0.524413)*x13 + (-0.002365)*x14 + (-0.088702)
x24 = (0.039848)*x10 + (0.238913)*x11 + (0.809323)*x12 + (-0.026897)*x13 + (0.394291)*x14 + (0.155813)
#
ReLU(x20)
ReLU(x21)
ReLU(x22)
ReLU(x23)
ReLU(x24)

x30 = (0.193647)*x20 + (0.445014)*x21 + (-0.201509)*x22 + (-0.138734)*x23 + (-0.004272)*x24 + (-0.217030)
x31 = (0.359547)*x20 + (-0.411097)*x21 + (-0.219632)*x22 + (0.064873)*x23 + (-0.258491)*x24 + (-0.312948)
x32 = (0.341159)*x20 + (-0.029092)*x21 + (-0.115886)*x22 + (0.405846)*x23 + (0.619939)*x24 + (0.137400)
x33 = (0.870996)*x20 + (0.200394)*x21 + (-0.012230)*x22 + (0.593420)*x23 + (0.556429)*x24 + (-0.245056)
x34 = (-0.295732)*x20 + (0.203392)*x21 + (0.052403)*x22 + (-0.238267)*x23 + (-0.001799)*x24 + (0.786413)
#
ReLU(x30)
ReLU(x31)
ReLU(x32)
ReLU(x33)
ReLU(x34)

x40 = (0.032599)*x30 + (-0.433082)*x31 + (0.213909)*x32 + (-0.158265)*x33 + (0.399385)*x34 + (-0.374439)
x41 = (-0.338420)*x30 + (-0.104633)*x31 + (0.616423)*x32 + (0.443235)*x33 + (-0.349890)*x34 + (-0.161576)
x42 = (-0.387245)*x30 + (-0.004803)*x31 + (0.172764)*x32 + (-0.371275)*x33 + (-0.374230)*x34 + (-0.391686)
x43 = (-0.132287)*x30 + (-0.091718)*x31 + (0.169309)*x32 + (-0.676320)*x33 + (0.907626)*x34 + (0.971672)
x44 = (-0.346559)*x30 + (-0.065136)*x31 + (-0.228492)*x32 + (0.140747)*x33 + (-0.374119)*x34 + (-0.064001)
#
ReLU(x40)
ReLU(x41)
ReLU(x42)
ReLU(x43)
ReLU(x44)

x50 = (-0.249582)*x40 + (0.451271)*x41 + (-0.101462)*x42 + (-0.176400)*x43 + (-0.138631)*x44 + (0.112350)
x51 = (-0.372951)*x40 + (0.015306)*x41 + (0.278476)*x42 + (-0.442605)*x43 + (0.046553)*x44 + (-0.077367)
x52 = (0.085866)*x40 + (0.328428)*x41 + (-0.011067)*x42 + (-0.453087)*x43 + (-0.347188)*x44 + (0.238621)
x53 = (0.072943)*x40 + (-0.389162)*x41 + (-0.369509)*x42 + (1.106869)*x43 + (-0.306143)*x44 + (0.526527)
x54 = (0.133786)*x40 + (0.596178)*x41 + (0.347561)*x42 + (-0.715858)*x43 + (0.243819)*x44 + (0.346449)
#
ReLU(x50)
ReLU(x51)
ReLU(x52)
ReLU(x53)
ReLU(x54)

x60 = (0.223427)*x50 + (0.140883)*x51 + (-0.361794)*x52 + (0.174790)*x53 + (-0.412156)*x54 + (-0.016673)
x61 = (0.383825)*x50 + (-0.058310)*x51 + (0.208050)*x52 + (-0.841193)*x53 + (0.180229)*x54 + (-0.398045)
#