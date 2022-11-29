
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


x10 = (-0.086983)*x00 + (-0.109302)*x01 + (-0.179455)*x02 + (0.044582)*x03 + (0.166409)*x04 + (0.111689)*x05 + (0.221887)*x06 + (0.129789)*x07 + (-0.105012)*x08 + (0.142727)*x09 + (0.079914)*x010 + (0.176220)*x011 + (-0.176501)*x012 + (0.245581)*x013 + (-0.085859)*x014 + (-0.011404)*x015 + (0.135192)*x016 + (-0.094529)*x017 + (-0.075012)*x018 + (-0.007535)*x019 + (-0.181592)*x020 + (-0.137313)*x021 + (0.178539)*x022 + (0.221554)
x11 = (0.085555)*x00 + (0.180593)*x01 + (0.095973)*x02 + (0.145634)*x03 + (-0.013054)*x04 + (-0.046131)*x05 + (-0.101860)*x06 + (0.087504)*x07 + (0.144026)*x08 + (-0.001171)*x09 + (0.019821)*x010 + (-0.079244)*x011 + (-0.084026)*x012 + (0.094073)*x013 + (-0.147142)*x014 + (-0.091479)*x015 + (-0.123430)*x016 + (0.086012)*x017 + (-0.130400)*x018 + (-0.042945)*x019 + (-0.133735)*x020 + (0.094505)*x021 + (-0.140885)*x022 + (-0.170776)
x12 = (-0.190092)*x00 + (-0.209266)*x01 + (-0.191580)*x02 + (-0.069748)*x03 + (-0.105046)*x04 + (-0.017809)*x05 + (-0.095893)*x06 + (0.094423)*x07 + (-0.156615)*x08 + (-0.081641)*x09 + (0.162829)*x010 + (0.035390)*x011 + (0.108297)*x012 + (-0.019451)*x013 + (0.022148)*x014 + (-0.205463)*x015 + (0.032041)*x016 + (-0.124331)*x017 + (-0.137138)*x018 + (-0.146897)*x019 + (-0.164085)*x020 + (-0.000478)*x021 + (0.121057)*x022 + (0.192262)
x13 = (-0.201735)*x00 + (-0.099441)*x01 + (-0.084256)*x02 + (0.121880)*x03 + (-0.126501)*x04 + (0.107950)*x05 + (-0.076340)*x06 + (0.189672)*x07 + (0.191301)*x08 + (0.092576)*x09 + (0.057785)*x010 + (-0.183400)*x011 + (0.069400)*x012 + (0.201617)*x013 + (-0.048941)*x014 + (0.016520)*x015 + (-0.110774)*x016 + (-0.101530)*x017 + (-0.185191)*x018 + (-0.120975)*x019 + (-0.141256)*x020 + (-0.007617)*x021 + (-0.125940)*x022 + (-0.075015)
x14 = (0.132004)*x00 + (0.164936)*x01 + (-0.194558)*x02 + (0.056716)*x03 + (-0.095749)*x04 + (0.181872)*x05 + (-0.109884)*x06 + (-0.025670)*x07 + (0.174060)*x08 + (0.106737)*x09 + (-0.209792)*x010 + (-0.151422)*x011 + (-0.116707)*x012 + (0.071330)*x013 + (-0.174262)*x014 + (0.116269)*x015 + (0.113748)*x016 + (-0.136969)*x017 + (0.096119)*x018 + (-0.104822)*x019 + (0.040556)*x020 + (0.199835)*x021 + (0.094941)*x022 + (0.179362)
#
ReLU(x10)
ReLU(x11)
ReLU(x12)
ReLU(x13)
ReLU(x14)

x20 = (0.217347)*x10 + (0.381804)*x11 + (-0.362736)*x12 + (-0.320985)*x13 + (-0.047861)*x14 + (-0.156343)
x21 = (-0.241365)*x10 + (-0.099622)*x11 + (-0.190272)*x12 + (0.425049)*x13 + (-0.338408)*x14 + (0.269818)
x22 = (0.057737)*x10 + (0.013291)*x11 + (-0.427613)*x12 + (-0.437579)*x13 + (0.033280)*x14 + (-0.120457)
x23 = (-0.344368)*x10 + (-0.122958)*x11 + (-0.123283)*x12 + (0.062613)*x13 + (0.176672)*x14 + (-0.356769)
x24 = (0.417187)*x10 + (-0.267922)*x11 + (0.424163)*x12 + (-0.430165)*x13 + (-0.303981)*x14 + (0.068911)
#
ReLU(x20)
ReLU(x21)
ReLU(x22)
ReLU(x23)
ReLU(x24)

x30 = (0.214245)*x20 + (0.497618)*x21 + (-0.205733)*x22 + (0.304465)*x23 + (-0.395151)*x24 + (0.153566)
x31 = (0.194093)*x20 + (-0.290545)*x21 + (-0.158573)*x22 + (0.399588)*x23 + (0.238940)*x24 + (-0.189964)
x32 = (-0.040121)*x20 + (-0.185584)*x21 + (-0.009339)*x22 + (0.107156)*x23 + (0.376375)*x24 + (-0.386102)
x33 = (-0.090850)*x20 + (-0.031298)*x21 + (-0.184134)*x22 + (-0.140794)*x23 + (-0.166174)*x24 + (0.212290)
x34 = (0.174741)*x20 + (-0.199683)*x21 + (0.391475)*x22 + (-0.085404)*x23 + (-0.346766)*x24 + (-0.290673)
#
ReLU(x30)
ReLU(x31)
ReLU(x32)
ReLU(x33)
ReLU(x34)

x40 = (-0.373190)*x30 + (-0.440493)*x31 + (-0.291836)*x32 + (0.166647)*x33 + (-0.368261)*x34 + (-0.041870)
x41 = (-0.091050)*x30 + (0.270413)*x31 + (-0.266405)*x32 + (0.119015)*x33 + (0.444470)*x34 + (0.133621)
x42 = (0.525719)*x30 + (-0.312905)*x31 + (-0.140353)*x32 + (0.121003)*x33 + (0.217578)*x34 + (0.367142)
x43 = (-0.070280)*x30 + (0.001394)*x31 + (-0.097111)*x32 + (-0.319863)*x33 + (-0.227547)*x34 + (-0.388337)
x44 = (0.264958)*x30 + (-0.268917)*x31 + (0.157814)*x32 + (0.432012)*x33 + (-0.293207)*x34 + (-0.309859)
#
ReLU(x40)
ReLU(x41)
ReLU(x42)
ReLU(x43)
ReLU(x44)

x50 = (0.010110)*x40 + (0.105485)*x41 + (-0.175320)*x42 + (0.274979)*x43 + (-0.392845)*x44 + (-0.276303)
x51 = (-0.422956)*x40 + (-0.406645)*x41 + (-0.385342)*x42 + (0.354918)*x43 + (0.412399)*x44 + (-0.312258)
x52 = (-0.282137)*x40 + (0.114048)*x41 + (-0.276050)*x42 + (0.108845)*x43 + (-0.103145)*x44 + (-0.412882)
x53 = (-0.330658)*x40 + (0.326737)*x41 + (-0.175657)*x42 + (0.189809)*x43 + (0.082198)*x44 + (0.178732)
x54 = (-0.121767)*x40 + (-0.199851)*x41 + (-0.187526)*x42 + (-0.295221)*x43 + (0.277918)*x44 + (0.108390)
#
ReLU(x50)
ReLU(x51)
ReLU(x52)
ReLU(x53)
ReLU(x54)

x60 = (0.249945)*x50 + (-0.030273)*x51 + (0.330564)*x52 + (-0.047829)*x53 + (-0.232771)*x54 + (-0.097393)
x61 = (-0.008528)*x50 + (0.097545)*x51 + (0.240443)*x52 + (-0.213437)*x53 + (-0.327162)*x54 + (-0.384804)
#