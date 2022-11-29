
#
assume(x03 <= 0.5)
assume(0 <= x04 <= 0)
assume(1 <= x05 <= 1)
assume(0 <= x06 <= 0)
assume(1 <= x07 <= 1)
assume(0 <= x08 <= 0)
assume(0 <= x09 <= 0)
assume(x10 <= 0.5)
assume(1 <= x11 <= 1)
assume(0 <= x12 <= 0)
assume(1 <= x13 <= 1)
assume(0 <= x14 <= 0)
assume(x15 <= 0.5)
assume(1 <= x18 <= 1)
assume(0 <= x19 <= 0)
assume(0 <= x20 <= 0)
assume(x21 <= 0.5)
assume(x22 <= 0.5)


x10 = (0.300846)*x00 + (0.124387)*x01 + (-0.112455)*x02 + (0.045906)*x03 + (-0.083341)*x04 + (0.182942)*x05 + (0.237110)*x06 + (0.088194)*x07 + (0.243231)*x08 + (0.153064)*x09 + (-0.397241)*x010 + (0.952976)*x011 + (-0.735339)*x012 + (-0.252922)*x013 + (0.132505)*x014 + (0.035504)*x015 + (0.128005)*x016 + (-0.037017)*x017 + (0.030004)*x018 + (0.231287)*x019 + (0.076495)*x020 + (0.031828)*x021 + (-0.007808)*x022 + (-0.118633)
x11 = (0.162843)*x00 + (0.009382)*x01 + (0.123381)*x02 + (-0.051523)*x03 + (-0.336327)*x04 + (0.226688)*x05 + (0.414868)*x06 + (-0.138045)*x07 + (0.083364)*x08 + (0.074893)*x09 + (-0.187638)*x010 + (1.157007)*x011 + (-0.759190)*x012 + (-0.085633)*x013 + (0.078220)*x014 + (0.171749)*x015 + (0.028522)*x016 + (0.062026)*x017 + (0.160086)*x018 + (0.148032)*x019 + (-0.159231)*x020 + (0.101991)*x021 + (0.050565)*x022 + (0.010258)
x12 = (0.026593)*x00 + (0.011730)*x01 + (0.199897)*x02 + (0.014564)*x03 + (-0.254781)*x04 + (0.095946)*x05 + (0.040668)*x06 + (0.037960)*x07 + (-0.172146)*x08 + (0.022242)*x09 + (-0.287190)*x010 + (0.309687)*x011 + (-0.533008)*x012 + (-0.018530)*x013 + (0.272790)*x014 + (-0.230443)*x015 + (0.200827)*x016 + (0.117598)*x017 + (-0.118579)*x018 + (0.225770)*x019 + (-0.053965)*x020 + (-0.208008)*x021 + (-0.013870)*x022 + (-0.025800)
x13 = (0.043560)*x00 + (-0.085135)*x01 + (0.070963)*x02 + (0.073358)*x03 + (0.084836)*x04 + (-0.130718)*x05 + (-0.181347)*x06 + (-0.029460)*x07 + (-0.195632)*x08 + (0.043482)*x09 + (-0.202962)*x010 + (-0.037279)*x011 + (0.161764)*x012 + (-0.140016)*x013 + (-0.049942)*x014 + (0.086038)*x015 + (-0.157689)*x016 + (-0.197495)*x017 + (-0.079469)*x018 + (0.045362)*x019 + (0.054308)*x020 + (-0.121786)*x021 + (-0.051936)*x022 + (0.007908)
x14 = (-0.058698)*x00 + (-0.129551)*x01 + (0.005544)*x02 + (-0.059248)*x03 + (-0.191414)*x04 + (-0.136357)*x05 + (0.257914)*x06 + (0.078390)*x07 + (-0.204942)*x08 + (0.137277)*x09 + (-0.279336)*x010 + (0.075310)*x011 + (-0.212100)*x012 + (-0.024693)*x013 + (-0.153690)*x014 + (-0.004425)*x015 + (0.302732)*x016 + (-0.032466)*x017 + (0.183910)*x018 + (0.109501)*x019 + (0.130606)*x020 + (-0.125227)*x021 + (0.129817)*x022 + (-0.003809)
#
ReLU(x10)
ReLU(x11)
ReLU(x12)
ReLU(x13)
ReLU(x14)

x20 = (0.201439)*x10 + (-0.404566)*x11 + (0.249953)*x12 + (0.158711)*x13 + (-0.218036)*x14 + (-0.303252)
x21 = (0.238783)*x10 + (-0.002084)*x11 + (-0.391941)*x12 + (0.408411)*x13 + (0.169046)*x14 + (-0.218853)
x22 = (0.181926)*x10 + (-0.064939)*x11 + (0.040612)*x12 + (0.083348)*x13 + (0.345432)*x14 + (-0.484521)
x23 = (1.033631)*x10 + (1.032815)*x11 + (0.321031)*x12 + (-0.306855)*x13 + (0.394705)*x14 + (-0.164121)
x24 = (-0.125221)*x10 + (0.210946)*x11 + (0.297805)*x12 + (-0.168131)*x13 + (-0.003529)*x14 + (0.203026)
#
ReLU(x20)
ReLU(x21)
ReLU(x22)
ReLU(x23)
ReLU(x24)

x30 = (-0.365378)*x20 + (-0.235096)*x21 + (-0.355946)*x22 + (-0.188929)*x23 + (0.279662)*x24 + (-0.199732)
x31 = (0.272905)*x20 + (0.455497)*x21 + (0.163089)*x22 + (0.647377)*x23 + (-0.250174)*x24 + (-0.204163)
x32 = (-0.201693)*x20 + (-0.347612)*x21 + (-0.445801)*x22 + (0.650096)*x23 + (0.024303)*x24 + (0.318890)
x33 = (-0.330749)*x20 + (0.010014)*x21 + (-0.398321)*x22 + (1.066950)*x23 + (0.253776)*x24 + (-0.249626)
x34 = (-0.052837)*x20 + (-0.028621)*x21 + (0.032252)*x22 + (-0.232565)*x23 + (0.163871)*x24 + (0.530165)
#
ReLU(x30)
ReLU(x31)
ReLU(x32)
ReLU(x33)
ReLU(x34)

x40 = (-0.143419)*x30 + (0.378626)*x31 + (0.261471)*x32 + (-0.327377)*x33 + (-0.407723)*x34 + (-0.276633)
x41 = (0.208978)*x30 + (-0.482260)*x31 + (0.549341)*x32 + (-0.477734)*x33 + (0.624225)*x34 + (0.612450)
x42 = (-0.020745)*x30 + (0.572898)*x31 + (0.715419)*x32 + (1.073481)*x33 + (-0.315372)*x34 + (0.265383)
x43 = (-0.200867)*x30 + (0.123139)*x31 + (-0.034413)*x32 + (-0.355498)*x33 + (-0.442025)*x34 + (-0.235176)
x44 = (-0.312211)*x30 + (0.241930)*x31 + (-0.068572)*x32 + (-0.040996)*x33 + (-0.264681)*x34 + (-0.356808)
#
ReLU(x40)
ReLU(x41)
ReLU(x42)
ReLU(x43)
ReLU(x44)

x50 = (0.107053)*x40 + (-0.002099)*x41 + (-0.139229)*x42 + (-0.443982)*x43 + (0.395980)*x44 + (-0.244034)
x51 = (-0.122706)*x40 + (0.162670)*x41 + (-0.364923)*x42 + (-0.367097)*x43 + (0.300717)*x44 + (-0.087408)
x52 = (-0.240924)*x40 + (-0.660522)*x41 + (0.465599)*x42 + (0.074167)*x43 + (-0.327023)*x44 + (0.139956)
x53 = (0.298728)*x40 + (-0.190767)*x41 + (0.292231)*x42 + (0.335927)*x43 + (-0.321029)*x44 + (-0.156095)
x54 = (0.050435)*x40 + (0.176278)*x41 + (-0.168843)*x42 + (0.291639)*x43 + (0.364444)*x44 + (-0.446010)
#
ReLU(x50)
ReLU(x51)
ReLU(x52)
ReLU(x53)
ReLU(x54)

x60 = (0.433229)*x50 + (-0.056054)*x51 + (0.550315)*x52 + (0.368157)*x53 + (-0.039155)*x54 + (-0.842954)
x61 = (-0.264976)*x50 + (0.120108)*x51 + (-0.465500)*x52 + (-0.028007)*x53 + (-0.372030)*x54 + (0.639474)
#
