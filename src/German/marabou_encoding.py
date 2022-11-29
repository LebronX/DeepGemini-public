from maraboupy import Marabou
from maraboupy import MarabouCore
import sys

sys.path.append("/Users/xiexuan/Downloads/Tool/Marabou")


class marabouEncoding:
    def __init__(self):
        self.var = {}

    def checkProperties(self, prop, networkFile, query):
        # Reading DNN using our own version of reading onnx file
        self.network_verified = Marabou.read_onnx_deepproperty(networkFile)
        self.inputVars_verified = self.network_verified.inputVars  # 2*(8and8)
        self.outputVars_verified = self.network_verified.outputVars  # 2*(2and2)

        if prop[0] == "checking-fairness":
            print("-----------checking fairness----------")
            return self.checkFair()
        elif prop[0] == "CEGFA":
            print("-----------Counter-Example Guided Fairness Analysis----------")
            if query == 0 or query == 1:
                return self.checkFairRegionQuery1(prop[1])
            elif query == 2:
                return self.checkFairRegionQuery2(prop[1])
            elif query == 3:
                return self.checkFairRegionQuery3(prop[1])

    def checkFair(self):

        # Assume for whole space

        #####################
        ### Pre-condition ###
        #####################
        for i in range(len(self.inputVars_verified)):
            for j in range(len(self.inputVars_verified[0])):
                if 0 <= j <= 2:  # account,3
                    self.network_verified.setLowerBound(
                        self.inputVars_verified[i][j], 0
                    )
                    self.network_verified.setUpperBound(
                        self.inputVars_verified[i][j], 1
                    )
                elif 3 <= j <= 4:  # credit history, 2
                    self.network_verified.setLowerBound(
                        self.inputVars_verified[i][j], 0
                    )
                    self.network_verified.setUpperBound(
                        self.inputVars_verified[i][j], 1
                    )
                elif j == 5:  # credit amount, 1
                    self.network_verified.setLowerBound(
                        self.inputVars_verified[i][j], 0
                    )
                    self.network_verified.setUpperBound(
                        self.inputVars_verified[i][j], 1
                    )
                elif 6 <= j <= 8:  # saving, 3
                    self.network_verified.setLowerBound(
                        self.inputVars_verified[i][j], 0
                    )
                    self.network_verified.setUpperBound(
                        self.inputVars_verified[i][j], 1
                    )
                elif 9 <= j <= 11:  # employment,3
                    self.network_verified.setLowerBound(
                        self.inputVars_verified[i][j], 0
                    )
                    self.network_verified.setUpperBound(
                        self.inputVars_verified[i][j], 1
                    )
                elif 12 <= j <= 13:  # gender,2
                    self.network_verified.setLowerBound(
                        self.inputVars_verified[i][j], 0
                    )
                    self.network_verified.setUpperBound(
                        self.inputVars_verified[i][j], 1
                    )
                elif j == 14:  # residence, 1
                    self.network_verified.setLowerBound(
                        self.inputVars_verified[i][j], 0
                    )
                    self.network_verified.setUpperBound(
                        self.inputVars_verified[i][j], 100
                    )
                elif 15 <= j <= 16:  # house, 2
                    self.network_verified.setLowerBound(
                        self.inputVars_verified[i][j], 0
                    )
                    self.network_verified.setUpperBound(
                        self.inputVars_verified[i][j], 1
                    )

        # One-hot encoding constraint
        for i in range(len(self.inputVars_verified)):
            for j in range(len(self.inputVars_verified[0])):
                # For 3 variables
                # Original: (a = 1 and b = 0 and c = 0) or
                #           (a = 0 and b = 1 and c = 0) or
                #           (a = 0 and b = 0 and c = 1)
                # Using a|(b&c) = (a|b)&(a|c)
                # Finally: (a = 0 or b = 0) and
                #          (a = 0 or c = 0) and
                #          (b = 0 or c = 0) and
                #          (a = 0 or b = 0 or c = 1) and
                #          (a = 1 or b = 1 or c = 1)
                if j == [0, 6, 9]:
                    # (a = 0 or b = 0) and
                    disjunction_pre_02 = []
                    eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                    eq_pre_01.addAddend(1, self.inputVars_verified[i][j])
                    eq_pre_01.setScalar(0)
                    disjunction_pre_02.append([eq_pre_01])
                    eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                    eq_pre_02.addAddend(1, self.inputVars_verified[i][j + 1])
                    eq_pre_02.setScalar(0)
                    disjunction_pre_02.append([eq_pre_02])
                    self.network_verified.addDisjunctionConstraint(disjunction_pre_02)

                    # (a = 0 or c = 0) and
                    disjunction_pre_02 = []
                    eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                    eq_pre_01.addAddend(1, self.inputVars_verified[i][j])
                    eq_pre_01.setScalar(0)
                    disjunction_pre_02.append([eq_pre_01])
                    eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                    eq_pre_02.addAddend(1, self.inputVars_verified[i][j + 2])
                    eq_pre_02.setScalar(0)
                    disjunction_pre_02.append([eq_pre_02])
                    self.network_verified.addDisjunctionConstraint(disjunction_pre_02)

                    # (b = 0 or c = 0) and
                    disjunction_pre_02 = []
                    eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                    eq_pre_01.addAddend(1, self.inputVars_verified[i][j + 1])
                    eq_pre_01.setScalar(0)
                    disjunction_pre_02.append([eq_pre_01])
                    eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                    eq_pre_02.addAddend(1, self.inputVars_verified[i][j + 2])
                    eq_pre_02.setScalar(0)
                    disjunction_pre_02.append([eq_pre_02])
                    self.network_verified.addDisjunctionConstraint(disjunction_pre_02)

                    # (a = 0 or b = 0 or c = 1) and
                    disjunction_pre_02 = []
                    eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                    eq_pre_01.addAddend(1, self.inputVars_verified[i][j])
                    eq_pre_01.setScalar(0)
                    disjunction_pre_02.append([eq_pre_01])
                    eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                    eq_pre_02.addAddend(1, self.inputVars_verified[i][j + 1])
                    eq_pre_02.setScalar(0)
                    disjunction_pre_02.append([eq_pre_02])
                    eq_pre_03 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                    eq_pre_03.addAddend(1, self.inputVars_verified[i][j + 2])
                    eq_pre_03.setScalar(1)
                    disjunction_pre_02.append([eq_pre_03])
                    self.network_verified.addDisjunctionConstraint(disjunction_pre_02)

                    # (a = 1 or b = 1 or c = 1)
                    disjunction_pre_02 = []
                    eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                    eq_pre_01.addAddend(1, self.inputVars_verified[i][j])
                    eq_pre_01.setScalar(1)
                    disjunction_pre_02.append([eq_pre_01])
                    eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                    eq_pre_02.addAddend(1, self.inputVars_verified[i][j + 1])
                    eq_pre_02.setScalar(1)
                    disjunction_pre_02.append([eq_pre_02])
                    eq_pre_03 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                    eq_pre_03.addAddend(1, self.inputVars_verified[i][j + 2])
                    eq_pre_03.setScalar(1)
                    disjunction_pre_02.append([eq_pre_03])
                    self.network_verified.addDisjunctionConstraint(disjunction_pre_02)
                # For 2 variables
                elif j in [3, 12, 15]:
                    disjunction_pre_02 = []
                    eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                    eq_pre_01.addAddend(1, self.inputVars_verified[i][j])
                    eq_pre_01.setScalar(0)
                    disjunction_pre_02.append([eq_pre_01])
                    eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                    eq_pre_02.addAddend(1, self.inputVars_verified[i][j + 1])
                    eq_pre_02.setScalar(0)
                    disjunction_pre_02.append([eq_pre_02])
                    self.network_verified.addDisjunctionConstraint(disjunction_pre_02)

                    disjunction_pre_02 = []
                    eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                    eq_pre_01.addAddend(1, self.inputVars_verified[i][j])
                    eq_pre_01.setScalar(1)
                    disjunction_pre_02.append([eq_pre_01])
                    eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                    eq_pre_02.addAddend(1, self.inputVars_verified[i][j + 1])
                    eq_pre_02.setScalar(1)
                    disjunction_pre_02.append([eq_pre_02])
                    self.network_verified.addDisjunctionConstraint(disjunction_pre_02)

        # Marabou constraint is conjunction of disjunction (CNF)
        # Non-sensitive feature be the same
        for j in range(0, len(self.inputVars_verified[0])):
            if j == 12 or j == 13:
                continue  # Gender should be different
            disjunction_pre_1 = []
            eq_pre_1 = MarabouCore.Equation(MarabouCore.Equation.EQ)
            eq_pre_1.addAddend(1, self.inputVars_verified[0][j])
            eq_pre_1.addAddend(-1, self.inputVars_verified[1][j])
            eq_pre_1.setScalar(0)
            disjunction_pre_1.append([eq_pre_1])
            self.network_verified.addDisjunctionConstraint(disjunction_pre_1)

        # Sensitive feature different, gender should be different, one hot encoding
        disjunction_pre_21 = []
        disjunction_pre_22 = []
        sen_pre_11 = MarabouCore.Equation(MarabouCore.Equation.EQ)
        sen_pre_11.addAddend(1, self.inputVars_verified[0][12])
        sen_pre_11.setScalar(0)
        sen_pre_12 = MarabouCore.Equation(MarabouCore.Equation.EQ)
        sen_pre_12.addAddend(1, self.inputVars_verified[0][13])
        sen_pre_12.setScalar(1)
        disjunction_pre_21.append([sen_pre_11])
        disjunction_pre_22.append([sen_pre_12])
        self.network_verified.addDisjunctionConstraint(disjunction_pre_21)
        self.network_verified.addDisjunctionConstraint(disjunction_pre_22)

        disjunction_pre_31 = []
        disjunction_pre_32 = []
        sen_pre_21 = MarabouCore.Equation(MarabouCore.Equation.EQ)
        sen_pre_21.addAddend(1, self.inputVars_verified[1][12])
        sen_pre_21.setScalar(1)
        sen_pre_22 = MarabouCore.Equation(MarabouCore.Equation.EQ)
        sen_pre_22.addAddend(1, self.inputVars_verified[1][13])
        sen_pre_22.setScalar(0)
        disjunction_pre_31.append([sen_pre_21])
        disjunction_pre_32.append([sen_pre_22])
        self.network_verified.addDisjunctionConstraint(disjunction_pre_31)
        self.network_verified.addDisjunctionConstraint(disjunction_pre_32)

        #####################
        ### Post-condition ##
        #####################
        # Leverage a|(b&c) = (a|b)&(a|c)
        # It becomes (N1 = 0 or N2 = 0) and (N1 = 1 or N2 = 1)
        disjunction_post_1 = []
        diff_post_1 = MarabouCore.Equation(MarabouCore.Equation.GE)
        diff_post_1.addAddend(1, self.outputVars_verified[0][0])
        diff_post_1.addAddend(-1, self.outputVars_verified[0][1])
        diff_post_1.setScalar(0)
        diff_post_2 = MarabouCore.Equation(MarabouCore.Equation.GE)
        diff_post_2.addAddend(1, self.outputVars_verified[1][0])
        diff_post_2.addAddend(-1, self.outputVars_verified[1][1])
        diff_post_2.setScalar(0)
        disjunction_post_1.append([diff_post_1])
        disjunction_post_1.append([diff_post_2])

        disjunction_post_2 = []
        diff_post_3 = MarabouCore.Equation(MarabouCore.Equation.GE)
        diff_post_3.addAddend(1, self.outputVars_verified[0][1])
        diff_post_3.addAddend(-1, self.outputVars_verified[0][0])
        diff_post_3.setScalar(0)
        diff_post_4 = MarabouCore.Equation(MarabouCore.Equation.GE)
        diff_post_4.addAddend(1, self.outputVars_verified[1][1])
        diff_post_4.addAddend(-1, self.outputVars_verified[1][0])
        diff_post_4.setScalar(0)
        disjunction_post_2.append([diff_post_3])
        disjunction_post_2.append([diff_post_4])

        self.network_verified.addDisjunctionConstraint(disjunction_post_1)
        self.network_verified.addDisjunctionConstraint(disjunction_post_2)

        vals, stats = self.network_verified.solve()

        return vals

    def checkFairRegionQuery1(self, region):

        # print(region)
        for i in range(len(self.inputVars_verified)):
            for j in range(len(self.inputVars_verified[0])):
                if j == 5:
                    self.network_verified.setLowerBound(
                        self.inputVars_verified[i][j], region[2][0]
                    )
                    self.network_verified.setUpperBound(
                        self.inputVars_verified[i][j], region[2][1]
                    )
                elif j == 14:
                    self.network_verified.setLowerBound(
                        self.inputVars_verified[i][j], region[6][0]
                    )
                    self.network_verified.setUpperBound(
                        self.inputVars_verified[i][j], region[6][1]
                    )
                else:
                    self.network_verified.setLowerBound(
                        self.inputVars_verified[i][j], 0
                    )
                    self.network_verified.setUpperBound(
                        self.inputVars_verified[i][j], 1
                    )

        ####################
        ## Pre-condition ###
        ####################
        for i in range(len(self.inputVars_verified)):
            for j, sub_region in zip([0, 3, 5, 6, 9, 12, 14, 15], region):
                if j == 0 or j == 6 or j == 9:  # Three variable
                    if not (-1 in sub_region):
                        # One-hot encoding constraint

                        # For 3 variables
                        # Original: (a = 1 and b = 0 and c = 0) or
                        #           (a = 0 and b = 1 and c = 0) or
                        #           (a = 0 and b = 0 and c = 1)
                        # Using a|(b&c) = (a|b)&(a|c)
                        # Finally: (a = 0 or b = 0) and
                        #          (a = 0 or c = 0) and
                        #          (b = 0 or c = 0) and
                        #          (a = 0 or b = 0 or c = 1) and
                        #          (a = 1 or b = 1 or c = 1)

                        # (a = 0 or b = 0) and
                        disjunction_onehot_three = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(1, self.inputVars_verified[i][j])
                        eq_pre_01.setScalar(0)
                        disjunction_onehot_three.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(1, self.inputVars_verified[i][j + 1])
                        eq_pre_02.setScalar(0)
                        disjunction_onehot_three.append([eq_pre_02])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_three
                        )

                        # (a = 0 or c = 0) and
                        disjunction_onehot_three = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(1, self.inputVars_verified[i][j])
                        eq_pre_01.setScalar(0)
                        disjunction_onehot_three.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(1, self.inputVars_verified[i][j + 2])
                        eq_pre_02.setScalar(0)
                        disjunction_onehot_three.append([eq_pre_02])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_three
                        )

                        # (b = 0 or c = 0) and
                        disjunction_onehot_three = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(1, self.inputVars_verified[i][j + 1])
                        eq_pre_01.setScalar(0)
                        disjunction_onehot_three.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(1, self.inputVars_verified[i][j + 2])
                        eq_pre_02.setScalar(0)
                        disjunction_onehot_three.append([eq_pre_02])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_three
                        )

                        # (a = 0 or b = 0 or c = 1) and
                        disjunction_onehot_three = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(1, self.inputVars_verified[i][j])
                        eq_pre_01.setScalar(0)
                        disjunction_onehot_three.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(1, self.inputVars_verified[i][j + 1])
                        eq_pre_02.setScalar(0)
                        disjunction_onehot_three.append([eq_pre_02])
                        eq_pre_03 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_03.addAddend(1, self.inputVars_verified[i][j + 2])
                        eq_pre_03.setScalar(1)
                        disjunction_onehot_three.append([eq_pre_03])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_three
                        )

                        # (a = 1 or b = 1 or c = 1)
                        disjunction_onehot_three = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(1, self.inputVars_verified[i][j])
                        eq_pre_01.setScalar(1)
                        disjunction_onehot_three.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(1, self.inputVars_verified[i][j + 1])
                        eq_pre_02.setScalar(1)
                        disjunction_onehot_three.append([eq_pre_02])
                        eq_pre_03 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_03.addAddend(1, self.inputVars_verified[i][j + 2])
                        eq_pre_03.setScalar(1)
                        disjunction_onehot_three.append([eq_pre_03])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_three
                        )
                    elif sub_region.count(-1) == 1:
                        digit1 = list(set([0, 1, 2]).intersection(set(sub_region)))[0]
                        digit2 = list(set([0, 1, 2]).intersection(set(sub_region)))[1]
                        remain_digit = list(set([0, 1, 2]) ^ set(sub_region))[
                            0
                        ]  # Third variable should be zero
                        # One-hot encoding constraint
                        disjunction_onehot_two = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(1, self.inputVars_verified[i][j + digit1])
                        eq_pre_01.setScalar(0)
                        disjunction_onehot_two.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(1, self.inputVars_verified[i][j + digit2])
                        eq_pre_02.setScalar(0)
                        disjunction_onehot_two.append([eq_pre_02])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_two
                        )

                        disjunction_onehot_two = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(1, self.inputVars_verified[i][j + digit1])
                        eq_pre_01.setScalar(1)
                        disjunction_onehot_two.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(1, self.inputVars_verified[i][j + digit2])
                        eq_pre_02.setScalar(1)
                        disjunction_onehot_two.append([eq_pre_02])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_two
                        )

                        disjunction_eq = []
                        sen_pre_11 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        sen_pre_11.addAddend(
                            1, self.inputVars_verified[i][j + remain_digit]
                        )
                        sen_pre_11.setScalar(0)
                        disjunction_eq.append([sen_pre_11])
                        self.network_verified.addDisjunctionConstraint(disjunction_eq)

                    elif sub_region.count(-1) == 2:
                        remain_digit1 = list(set([0, 1, 2]) ^ set(sub_region))[0]
                        remain_digit2 = list(set([0, 1, 2]) ^ set(sub_region))[1]
                        allow_digit = list(
                            set([0, 1, 2]).intersection(set(sub_region))
                        )[0]
                        disjunction_eq = []
                        sen_pre_11 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        sen_pre_11.addAddend(
                            1, self.inputVars_verified[i][j + allow_digit]
                        )
                        sen_pre_11.setScalar(1)
                        disjunction_eq.append([sen_pre_11])
                        self.network_verified.addDisjunctionConstraint(disjunction_eq)
                        disjunction_eq = []
                        sen_pre_11 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        sen_pre_11.addAddend(
                            1, self.inputVars_verified[i][j + remain_digit1]
                        )
                        sen_pre_11.setScalar(0)
                        disjunction_eq.append([sen_pre_11])
                        self.network_verified.addDisjunctionConstraint(disjunction_eq)
                        disjunction_eq = []
                        sen_pre_11 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        sen_pre_11.addAddend(
                            1, self.inputVars_verified[i][j + remain_digit2]
                        )
                        sen_pre_11.setScalar(0)
                        disjunction_eq.append([sen_pre_11])
                        self.network_verified.addDisjunctionConstraint(disjunction_eq)

                elif j == 3 or j == 12 or j == 15:  # Two variable
                    if not (-1 in sub_region):
                        # One-hot encoding constraint
                        disjunction_onehot_two = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(
                            1, self.inputVars_verified[i][j + sub_region[0]]
                        )
                        eq_pre_01.setScalar(0)
                        disjunction_onehot_two.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(
                            1, self.inputVars_verified[i][j + sub_region[1]]
                        )
                        eq_pre_02.setScalar(0)
                        disjunction_onehot_two.append([eq_pre_02])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_two
                        )

                        disjunction_onehot_two = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(
                            1, self.inputVars_verified[i][j + sub_region[0]]
                        )
                        eq_pre_01.setScalar(1)
                        disjunction_onehot_two.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(
                            1, self.inputVars_verified[i][j + sub_region[1]]
                        )
                        eq_pre_02.setScalar(1)
                        disjunction_onehot_two.append([eq_pre_02])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_two
                        )

                    elif -1 in sub_region:
                        # print(j)
                        # print(sub_region)
                        # print("haha")
                        allow_digit = list(set([0, 1]).intersection(set(sub_region)))[0]
                        remain_digit = list(set([0, 1]) ^ set(sub_region))[0]
                        disjunction_eq1 = []
                        sen_pre_11 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        sen_pre_11.addAddend(
                            1, self.inputVars_verified[i][j + allow_digit]
                        )
                        sen_pre_11.setScalar(1)
                        disjunction_eq1.append([sen_pre_11])
                        self.network_verified.addDisjunctionConstraint(disjunction_eq1)
                        disjunction_eq2 = []
                        sen_pre_12 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        sen_pre_12.addAddend(
                            1, self.inputVars_verified[i][j + remain_digit]
                        )
                        sen_pre_12.setScalar(0)
                        disjunction_eq2.append([sen_pre_12])
                        self.network_verified.addDisjunctionConstraint(disjunction_eq2)

        # Marabou constraint is conjunction of disjunction (CNF)
        # Non-sensitive feature be the same
        for i in range(0, len(self.inputVars_verified[0])):
            if i == 12 or i == 13:
                continue  # Gender should be different
            disjunction_pre_1 = []
            eq_pre_1 = MarabouCore.Equation(MarabouCore.Equation.EQ)
            eq_pre_1.addAddend(1, self.inputVars_verified[0][i])
            eq_pre_1.addAddend(-1, self.inputVars_verified[1][i])
            eq_pre_1.setScalar(0)
            disjunction_pre_1.append([eq_pre_1])
            self.network_verified.addDisjunctionConstraint(disjunction_pre_1)

        # Sensitive feature different, gender should be different, one hot encoding
        disjunction_pre_21 = []
        disjunction_pre_22 = []
        sen_pre_11 = MarabouCore.Equation(MarabouCore.Equation.EQ)
        sen_pre_11.addAddend(1, self.inputVars_verified[0][12])
        sen_pre_11.setScalar(0)
        sen_pre_12 = MarabouCore.Equation(MarabouCore.Equation.EQ)
        sen_pre_12.addAddend(1, self.inputVars_verified[0][13])
        sen_pre_12.setScalar(1)
        disjunction_pre_21.append([sen_pre_11])
        disjunction_pre_22.append([sen_pre_12])
        self.network_verified.addDisjunctionConstraint(disjunction_pre_21)
        self.network_verified.addDisjunctionConstraint(disjunction_pre_22)

        disjunction_pre_31 = []
        disjunction_pre_32 = []
        sen_pre_21 = MarabouCore.Equation(MarabouCore.Equation.EQ)
        sen_pre_21.addAddend(1, self.inputVars_verified[1][12])
        sen_pre_21.setScalar(1)
        sen_pre_22 = MarabouCore.Equation(MarabouCore.Equation.EQ)
        sen_pre_22.addAddend(1, self.inputVars_verified[1][13])
        sen_pre_22.setScalar(0)
        disjunction_pre_31.append([sen_pre_21])
        disjunction_pre_32.append([sen_pre_22])
        self.network_verified.addDisjunctionConstraint(disjunction_pre_31)
        self.network_verified.addDisjunctionConstraint(disjunction_pre_32)

        #####################
        ### Post-condition ##
        #####################
        # Leverage a|(b&c) = (a|b)&(a|c)
        # (N1 = 0 and N2 = 1) or (N1 = 1 and N2 = 0)
        # It becomes (N1 = 0 or N2 = 0) and (N1 = 1 or N2 = 1)
        disjunction_post_1 = []
        diff_post_1 = MarabouCore.Equation(MarabouCore.Equation.GE)
        diff_post_1.addAddend(1, self.outputVars_verified[0][0])
        diff_post_1.addAddend(-1, self.outputVars_verified[0][1])
        diff_post_1.setScalar(
            0
        )  # XXX implementation problem, only support 0 (easy to find CE.) or 0.01 (might be hard to find CE.)
        diff_post_2 = MarabouCore.Equation(MarabouCore.Equation.GE)
        diff_post_2.addAddend(1, self.outputVars_verified[1][0])
        diff_post_2.addAddend(-1, self.outputVars_verified[1][1])
        diff_post_2.setScalar(0)
        disjunction_post_1.append([diff_post_1])
        disjunction_post_1.append([diff_post_2])

        disjunction_post_2 = []
        diff_post_3 = MarabouCore.Equation(MarabouCore.Equation.GE)
        diff_post_3.addAddend(1, self.outputVars_verified[0][1])
        diff_post_3.addAddend(-1, self.outputVars_verified[0][0])
        diff_post_3.setScalar(0)
        diff_post_4 = MarabouCore.Equation(MarabouCore.Equation.GE)
        diff_post_4.addAddend(1, self.outputVars_verified[1][1])
        diff_post_4.addAddend(-1, self.outputVars_verified[1][0])
        diff_post_4.setScalar(0)
        disjunction_post_2.append([diff_post_3])
        disjunction_post_2.append([diff_post_4])

        self.network_verified.addDisjunctionConstraint(disjunction_post_1)
        self.network_verified.addDisjunctionConstraint(disjunction_post_2)

        vals, stats = self.network_verified.solve()

        return vals

    def checkFairRegionQuery2(self, region):

        # print(region)
        for i in range(len(self.inputVars_verified)):
            for j in range(len(self.inputVars_verified[0])):
                if j == 5:  # Quantitative
                    self.network_verified.setLowerBound(
                        self.inputVars_verified[i][j], region[2][0]
                    )
                    self.network_verified.setUpperBound(
                        self.inputVars_verified[i][j], region[2][1]
                    )
                elif j == 14:  # Quantitative
                    self.network_verified.setLowerBound(
                        self.inputVars_verified[i][j], region[6][0]
                    )
                    self.network_verified.setUpperBound(
                        self.inputVars_verified[i][j], region[6][1]
                    )
                else:
                    self.network_verified.setLowerBound(
                        self.inputVars_verified[i][j], 0
                    )
                    self.network_verified.setUpperBound(
                        self.inputVars_verified[i][j], 1
                    )

        ####################
        ## Pre-condition ###
        ####################
        for i in range(len(self.inputVars_verified)):
            for j, sub_region in zip([0, 3, 5, 6, 9, 12, 14, 15], region):
                if j == 0 or j == 6 or j == 9:  # Three variable
                    if not (-1 in sub_region):
                        # One-hot encoding constraint

                        # For 3 variables
                        # Original: (a = 1 and b = 0 and c = 0) or
                        #           (a = 0 and b = 1 and c = 0) or
                        #           (a = 0 and b = 0 and c = 1)
                        # Using a|(b&c) = (a|b)&(a|c)
                        # Finally: (a = 0 or b = 0) and
                        #          (a = 0 or c = 0) and
                        #          (b = 0 or c = 0) and
                        #          (a = 0 or b = 0 or c = 1) and
                        #          (a = 1 or b = 1 or c = 1)

                        # (a = 0 or b = 0) and
                        disjunction_onehot_three = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(1, self.inputVars_verified[i][j])
                        eq_pre_01.setScalar(0)
                        disjunction_onehot_three.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(1, self.inputVars_verified[i][j + 1])
                        eq_pre_02.setScalar(0)
                        disjunction_onehot_three.append([eq_pre_02])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_three
                        )

                        # (a = 0 or c = 0) and
                        disjunction_onehot_three = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(1, self.inputVars_verified[i][j])
                        eq_pre_01.setScalar(0)
                        disjunction_onehot_three.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(1, self.inputVars_verified[i][j + 2])
                        eq_pre_02.setScalar(0)
                        disjunction_onehot_three.append([eq_pre_02])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_three
                        )

                        # (b = 0 or c = 0) and
                        disjunction_onehot_three = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(1, self.inputVars_verified[i][j + 1])
                        eq_pre_01.setScalar(0)
                        disjunction_onehot_three.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(1, self.inputVars_verified[i][j + 2])
                        eq_pre_02.setScalar(0)
                        disjunction_onehot_three.append([eq_pre_02])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_three
                        )

                        # (a = 0 or b = 0 or c = 1) and
                        disjunction_onehot_three = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(1, self.inputVars_verified[i][j])
                        eq_pre_01.setScalar(0)
                        disjunction_onehot_three.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(1, self.inputVars_verified[i][j + 1])
                        eq_pre_02.setScalar(0)
                        disjunction_onehot_three.append([eq_pre_02])
                        eq_pre_03 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_03.addAddend(1, self.inputVars_verified[i][j + 2])
                        eq_pre_03.setScalar(1)
                        disjunction_onehot_three.append([eq_pre_03])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_three
                        )

                        # (a = 1 or b = 1 or c = 1)
                        disjunction_onehot_three = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(1, self.inputVars_verified[i][j])
                        eq_pre_01.setScalar(1)
                        disjunction_onehot_three.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(1, self.inputVars_verified[i][j + 1])
                        eq_pre_02.setScalar(1)
                        disjunction_onehot_three.append([eq_pre_02])
                        eq_pre_03 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_03.addAddend(1, self.inputVars_verified[i][j + 2])
                        eq_pre_03.setScalar(1)
                        disjunction_onehot_three.append([eq_pre_03])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_three
                        )
                    elif sub_region.count(-1) == 1:
                        digit1 = list(set([0, 1, 2]).intersection(set(sub_region)))[0]
                        digit2 = list(set([0, 1, 2]).intersection(set(sub_region)))[1]
                        remain_digit = list(set([0, 1, 2]) ^ set(sub_region))[
                            0
                        ]  # Third variable should be zero
                        # One-hot encoding constraint
                        disjunction_onehot_two = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(1, self.inputVars_verified[i][j + digit1])
                        eq_pre_01.setScalar(0)
                        disjunction_onehot_two.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(1, self.inputVars_verified[i][j + digit2])
                        eq_pre_02.setScalar(0)
                        disjunction_onehot_two.append([eq_pre_02])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_two
                        )

                        disjunction_onehot_two = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(1, self.inputVars_verified[i][j + digit1])
                        eq_pre_01.setScalar(1)
                        disjunction_onehot_two.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(1, self.inputVars_verified[i][j + digit2])
                        eq_pre_02.setScalar(1)
                        disjunction_onehot_two.append([eq_pre_02])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_two
                        )

                        disjunction_eq = []
                        sen_pre_11 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        sen_pre_11.addAddend(
                            1, self.inputVars_verified[i][j + remain_digit]
                        )
                        sen_pre_11.setScalar(0)
                        disjunction_eq.append([sen_pre_11])
                        self.network_verified.addDisjunctionConstraint(disjunction_eq)

                    elif sub_region.count(-1) == 2:
                        remain_digit1 = list(set([0, 1, 2]) ^ set(sub_region))[0]
                        remain_digit2 = list(set([0, 1, 2]) ^ set(sub_region))[1]
                        allow_digit = list(
                            set([0, 1, 2]).intersection(set(sub_region))
                        )[0]
                        disjunction_eq = []
                        sen_pre_11 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        sen_pre_11.addAddend(
                            1, self.inputVars_verified[i][j + allow_digit]
                        )
                        sen_pre_11.setScalar(1)
                        disjunction_eq.append([sen_pre_11])
                        self.network_verified.addDisjunctionConstraint(disjunction_eq)
                        disjunction_eq = []
                        sen_pre_11 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        sen_pre_11.addAddend(
                            1, self.inputVars_verified[i][j + remain_digit1]
                        )
                        sen_pre_11.setScalar(0)
                        disjunction_eq.append([sen_pre_11])
                        self.network_verified.addDisjunctionConstraint(disjunction_eq)
                        disjunction_eq = []
                        sen_pre_11 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        sen_pre_11.addAddend(
                            1, self.inputVars_verified[i][j + remain_digit2]
                        )
                        sen_pre_11.setScalar(0)
                        disjunction_eq.append([sen_pre_11])
                        self.network_verified.addDisjunctionConstraint(disjunction_eq)

                elif j == 3 or j == 12 or j == 15:  # Two variable
                    if not (-1 in sub_region):
                        # One-hot encoding constraint
                        disjunction_onehot_two = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(
                            1, self.inputVars_verified[i][j + sub_region[0]]
                        )
                        eq_pre_01.setScalar(0)
                        disjunction_onehot_two.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(
                            1, self.inputVars_verified[i][j + sub_region[1]]
                        )
                        eq_pre_02.setScalar(0)
                        disjunction_onehot_two.append([eq_pre_02])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_two
                        )

                        disjunction_onehot_two = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(
                            1, self.inputVars_verified[i][j + sub_region[0]]
                        )
                        eq_pre_01.setScalar(1)
                        disjunction_onehot_two.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(
                            1, self.inputVars_verified[i][j + sub_region[1]]
                        )
                        eq_pre_02.setScalar(1)
                        disjunction_onehot_two.append([eq_pre_02])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_two
                        )

                    elif -1 in sub_region:
                        # print(j)
                        # print(sub_region)
                        # print("haha")
                        allow_digit = list(set([0, 1]).intersection(set(sub_region)))[0]
                        remain_digit = list(set([0, 1]) ^ set(sub_region))[0]
                        disjunction_eq1 = []
                        sen_pre_11 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        sen_pre_11.addAddend(
                            1, self.inputVars_verified[i][j + allow_digit]
                        )
                        sen_pre_11.setScalar(1)
                        disjunction_eq1.append([sen_pre_11])
                        self.network_verified.addDisjunctionConstraint(disjunction_eq1)
                        disjunction_eq2 = []
                        sen_pre_12 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        sen_pre_12.addAddend(
                            1, self.inputVars_verified[i][j + remain_digit]
                        )
                        sen_pre_12.setScalar(0)
                        disjunction_eq2.append([sen_pre_12])
                        self.network_verified.addDisjunctionConstraint(disjunction_eq2)

        # Marabou constraint is conjunction of disjunction (CNF)
        # Non-sensitive feature be the same
        for i in range(0, len(self.inputVars_verified[0])):
            if i == 3 or i == 4:
                continue  # credit history should be different
            disjunction_pre_1 = []
            eq_pre_1 = MarabouCore.Equation(MarabouCore.Equation.EQ)
            eq_pre_1.addAddend(1, self.inputVars_verified[0][i])
            eq_pre_1.addAddend(-1, self.inputVars_verified[1][i])
            eq_pre_1.setScalar(0)
            disjunction_pre_1.append([eq_pre_1])
            self.network_verified.addDisjunctionConstraint(disjunction_pre_1)

        # Sensitive feature different, Age should be different, one hot encoding
        disjunction_pre_21 = []
        disjunction_pre_22 = []
        sen_pre_11 = MarabouCore.Equation(MarabouCore.Equation.EQ)
        sen_pre_11.addAddend(1, self.inputVars_verified[0][3])
        sen_pre_11.setScalar(0)
        sen_pre_12 = MarabouCore.Equation(MarabouCore.Equation.EQ)
        sen_pre_12.addAddend(1, self.inputVars_verified[0][4])
        sen_pre_12.setScalar(1)
        disjunction_pre_21.append([sen_pre_11])
        disjunction_pre_22.append([sen_pre_12])
        self.network_verified.addDisjunctionConstraint(disjunction_pre_21)
        self.network_verified.addDisjunctionConstraint(disjunction_pre_22)

        disjunction_pre_31 = []
        disjunction_pre_32 = []
        sen_pre_21 = MarabouCore.Equation(MarabouCore.Equation.EQ)
        sen_pre_21.addAddend(1, self.inputVars_verified[1][3])
        sen_pre_21.setScalar(1)
        sen_pre_22 = MarabouCore.Equation(MarabouCore.Equation.EQ)
        sen_pre_22.addAddend(1, self.inputVars_verified[1][4])
        sen_pre_22.setScalar(0)
        disjunction_pre_31.append([sen_pre_21])
        disjunction_pre_32.append([sen_pre_22])
        self.network_verified.addDisjunctionConstraint(disjunction_pre_31)
        self.network_verified.addDisjunctionConstraint(disjunction_pre_32)

        #####################
        ### Post-condition ##
        #####################
        # Leverage a|(b&c) = (a|b)&(a|c)
        # (N1 = 0 and N2 = 1) or (N1 = 1 and N2 = 0)
        # It becomes (N1 = 0 or N2 = 0) and (N1 = 1 or N2 = 1)
        disjunction_post_1 = []
        diff_post_1 = MarabouCore.Equation(MarabouCore.Equation.GE)
        diff_post_1.addAddend(1, self.outputVars_verified[0][0])
        diff_post_1.addAddend(-1, self.outputVars_verified[0][1])
        diff_post_1.setScalar(
            0
        )  # XXX implementation problem, only support 0 (easy to find CE.) or 0.01 (might be hard to find CE.)
        diff_post_2 = MarabouCore.Equation(MarabouCore.Equation.GE)
        diff_post_2.addAddend(1, self.outputVars_verified[1][0])
        diff_post_2.addAddend(-1, self.outputVars_verified[1][1])
        diff_post_2.setScalar(0)
        disjunction_post_1.append([diff_post_1])
        disjunction_post_1.append([diff_post_2])

        disjunction_post_2 = []
        diff_post_3 = MarabouCore.Equation(MarabouCore.Equation.GE)
        diff_post_3.addAddend(1, self.outputVars_verified[0][1])
        diff_post_3.addAddend(-1, self.outputVars_verified[0][0])
        diff_post_3.setScalar(0)
        diff_post_4 = MarabouCore.Equation(MarabouCore.Equation.GE)
        diff_post_4.addAddend(1, self.outputVars_verified[1][1])
        diff_post_4.addAddend(-1, self.outputVars_verified[1][0])
        diff_post_4.setScalar(0)
        disjunction_post_2.append([diff_post_3])
        disjunction_post_2.append([diff_post_4])

        self.network_verified.addDisjunctionConstraint(disjunction_post_1)
        self.network_verified.addDisjunctionConstraint(disjunction_post_2)

        vals, stats = self.network_verified.solve()

        return vals


    def checkFairRegionQuery3(self, region):

        # print(region)
        for i in range(len(self.inputVars_verified)):
            for j in range(len(self.inputVars_verified[0])):
                if j == 5:  # Quantitative
                    self.network_verified.setLowerBound(
                        self.inputVars_verified[i][j], region[2][0]
                    )
                    self.network_verified.setUpperBound(
                        self.inputVars_verified[i][j], region[2][1]
                    )
                elif j == 14:  # Quantitative
                    self.network_verified.setLowerBound(
                        self.inputVars_verified[i][j], region[6][0]
                    )
                    self.network_verified.setUpperBound(
                        self.inputVars_verified[i][j], region[6][1]
                    )
                else:
                    self.network_verified.setLowerBound(
                        self.inputVars_verified[i][j], 0
                    )
                    self.network_verified.setUpperBound(
                        self.inputVars_verified[i][j], 1
                    )

        ####################
        ## Pre-condition ###
        ####################
        for i in range(len(self.inputVars_verified)):
            for j, sub_region in zip([0, 3, 5, 6, 9, 12, 14, 15], region):
                if j == 0 or j == 6 or j == 9:  # Three variable
                    if not (-1 in sub_region):
                        # One-hot encoding constraint

                        # For 3 variables
                        # Original: (a = 1 and b = 0 and c = 0) or
                        #           (a = 0 and b = 1 and c = 0) or
                        #           (a = 0 and b = 0 and c = 1)
                        # Using a|(b&c) = (a|b)&(a|c)
                        # Finally: (a = 0 or b = 0) and
                        #          (a = 0 or c = 0) and
                        #          (b = 0 or c = 0) and
                        #          (a = 0 or b = 0 or c = 1) and
                        #          (a = 1 or b = 1 or c = 1)

                        # (a = 0 or b = 0) and
                        disjunction_onehot_three = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(1, self.inputVars_verified[i][j])
                        eq_pre_01.setScalar(0)
                        disjunction_onehot_three.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(1, self.inputVars_verified[i][j + 1])
                        eq_pre_02.setScalar(0)
                        disjunction_onehot_three.append([eq_pre_02])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_three
                        )

                        # (a = 0 or c = 0) and
                        disjunction_onehot_three = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(1, self.inputVars_verified[i][j])
                        eq_pre_01.setScalar(0)
                        disjunction_onehot_three.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(1, self.inputVars_verified[i][j + 2])
                        eq_pre_02.setScalar(0)
                        disjunction_onehot_three.append([eq_pre_02])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_three
                        )

                        # (b = 0 or c = 0) and
                        disjunction_onehot_three = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(1, self.inputVars_verified[i][j + 1])
                        eq_pre_01.setScalar(0)
                        disjunction_onehot_three.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(1, self.inputVars_verified[i][j + 2])
                        eq_pre_02.setScalar(0)
                        disjunction_onehot_three.append([eq_pre_02])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_three
                        )

                        # (a = 0 or b = 0 or c = 1) and
                        disjunction_onehot_three = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(1, self.inputVars_verified[i][j])
                        eq_pre_01.setScalar(0)
                        disjunction_onehot_three.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(1, self.inputVars_verified[i][j + 1])
                        eq_pre_02.setScalar(0)
                        disjunction_onehot_three.append([eq_pre_02])
                        eq_pre_03 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_03.addAddend(1, self.inputVars_verified[i][j + 2])
                        eq_pre_03.setScalar(1)
                        disjunction_onehot_three.append([eq_pre_03])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_three
                        )

                        # (a = 1 or b = 1 or c = 1)
                        disjunction_onehot_three = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(1, self.inputVars_verified[i][j])
                        eq_pre_01.setScalar(1)
                        disjunction_onehot_three.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(1, self.inputVars_verified[i][j + 1])
                        eq_pre_02.setScalar(1)
                        disjunction_onehot_three.append([eq_pre_02])
                        eq_pre_03 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_03.addAddend(1, self.inputVars_verified[i][j + 2])
                        eq_pre_03.setScalar(1)
                        disjunction_onehot_three.append([eq_pre_03])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_three
                        )
                    elif sub_region.count(-1) == 1:
                        digit1 = list(set([0, 1, 2]).intersection(set(sub_region)))[0]
                        digit2 = list(set([0, 1, 2]).intersection(set(sub_region)))[1]
                        remain_digit = list(set([0, 1, 2]) ^ set(sub_region))[
                            0
                        ]  # Third variable should be zero
                        # One-hot encoding constraint
                        disjunction_onehot_two = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(1, self.inputVars_verified[i][j + digit1])
                        eq_pre_01.setScalar(0)
                        disjunction_onehot_two.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(1, self.inputVars_verified[i][j + digit2])
                        eq_pre_02.setScalar(0)
                        disjunction_onehot_two.append([eq_pre_02])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_two
                        )

                        disjunction_onehot_two = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(1, self.inputVars_verified[i][j + digit1])
                        eq_pre_01.setScalar(1)
                        disjunction_onehot_two.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(1, self.inputVars_verified[i][j + digit2])
                        eq_pre_02.setScalar(1)
                        disjunction_onehot_two.append([eq_pre_02])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_two
                        )

                        disjunction_eq = []
                        sen_pre_11 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        sen_pre_11.addAddend(
                            1, self.inputVars_verified[i][j + remain_digit]
                        )
                        sen_pre_11.setScalar(0)
                        disjunction_eq.append([sen_pre_11])
                        self.network_verified.addDisjunctionConstraint(disjunction_eq)

                    elif sub_region.count(-1) == 2:
                        remain_digit1 = list(set([0, 1, 2]) ^ set(sub_region))[0]
                        remain_digit2 = list(set([0, 1, 2]) ^ set(sub_region))[1]
                        allow_digit = list(
                            set([0, 1, 2]).intersection(set(sub_region))
                        )[0]
                        disjunction_eq = []
                        sen_pre_11 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        sen_pre_11.addAddend(
                            1, self.inputVars_verified[i][j + allow_digit]
                        )
                        sen_pre_11.setScalar(1)
                        disjunction_eq.append([sen_pre_11])
                        self.network_verified.addDisjunctionConstraint(disjunction_eq)
                        disjunction_eq = []
                        sen_pre_11 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        sen_pre_11.addAddend(
                            1, self.inputVars_verified[i][j + remain_digit1]
                        )
                        sen_pre_11.setScalar(0)
                        disjunction_eq.append([sen_pre_11])
                        self.network_verified.addDisjunctionConstraint(disjunction_eq)
                        disjunction_eq = []
                        sen_pre_11 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        sen_pre_11.addAddend(
                            1, self.inputVars_verified[i][j + remain_digit2]
                        )
                        sen_pre_11.setScalar(0)
                        disjunction_eq.append([sen_pre_11])
                        self.network_verified.addDisjunctionConstraint(disjunction_eq)

                elif j == 3 or j == 12 or j == 15:  # Two variable
                    if not (-1 in sub_region):
                        # One-hot encoding constraint
                        disjunction_onehot_two = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(
                            1, self.inputVars_verified[i][j + sub_region[0]]
                        )
                        eq_pre_01.setScalar(0)
                        disjunction_onehot_two.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(
                            1, self.inputVars_verified[i][j + sub_region[1]]
                        )
                        eq_pre_02.setScalar(0)
                        disjunction_onehot_two.append([eq_pre_02])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_two
                        )

                        disjunction_onehot_two = []
                        eq_pre_01 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_01.addAddend(
                            1, self.inputVars_verified[i][j + sub_region[0]]
                        )
                        eq_pre_01.setScalar(1)
                        disjunction_onehot_two.append([eq_pre_01])
                        eq_pre_02 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq_pre_02.addAddend(
                            1, self.inputVars_verified[i][j + sub_region[1]]
                        )
                        eq_pre_02.setScalar(1)
                        disjunction_onehot_two.append([eq_pre_02])
                        self.network_verified.addDisjunctionConstraint(
                            disjunction_onehot_two
                        )

                    elif -1 in sub_region:
                        # print(j)
                        # print(sub_region)
                        # print("haha")
                        allow_digit = list(set([0, 1]).intersection(set(sub_region)))[0]
                        remain_digit = list(set([0, 1]) ^ set(sub_region))[0]
                        disjunction_eq1 = []
                        sen_pre_11 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        sen_pre_11.addAddend(
                            1, self.inputVars_verified[i][j + allow_digit]
                        )
                        sen_pre_11.setScalar(1)
                        disjunction_eq1.append([sen_pre_11])
                        self.network_verified.addDisjunctionConstraint(disjunction_eq1)
                        disjunction_eq2 = []
                        sen_pre_12 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        sen_pre_12.addAddend(
                            1, self.inputVars_verified[i][j + remain_digit]
                        )
                        sen_pre_12.setScalar(0)
                        disjunction_eq2.append([sen_pre_12])
                        self.network_verified.addDisjunctionConstraint(disjunction_eq2)

        # Marabou constraint is conjunction of disjunction (CNF)
        # Non-sensitive feature be the same
        for i in range(0, len(self.inputVars_verified[0])):
            if i == 15 or i == 16:
                continue  # credit history should be different
            disjunction_pre_1 = []
            eq_pre_1 = MarabouCore.Equation(MarabouCore.Equation.EQ)
            eq_pre_1.addAddend(1, self.inputVars_verified[0][i])
            eq_pre_1.addAddend(-1, self.inputVars_verified[1][i])
            eq_pre_1.setScalar(0)
            disjunction_pre_1.append([eq_pre_1])
            self.network_verified.addDisjunctionConstraint(disjunction_pre_1)

        # Sensitive feature different, Age should be different, one hot encoding
        disjunction_pre_21 = []
        disjunction_pre_22 = []
        sen_pre_11 = MarabouCore.Equation(MarabouCore.Equation.EQ)
        sen_pre_11.addAddend(1, self.inputVars_verified[0][15])
        sen_pre_11.setScalar(0)
        sen_pre_12 = MarabouCore.Equation(MarabouCore.Equation.EQ)
        sen_pre_12.addAddend(1, self.inputVars_verified[0][16])
        sen_pre_12.setScalar(1)
        disjunction_pre_21.append([sen_pre_11])
        disjunction_pre_22.append([sen_pre_12])
        self.network_verified.addDisjunctionConstraint(disjunction_pre_21)
        self.network_verified.addDisjunctionConstraint(disjunction_pre_22)

        disjunction_pre_31 = []
        disjunction_pre_32 = []
        sen_pre_21 = MarabouCore.Equation(MarabouCore.Equation.EQ)
        sen_pre_21.addAddend(1, self.inputVars_verified[1][15])
        sen_pre_21.setScalar(1)
        sen_pre_22 = MarabouCore.Equation(MarabouCore.Equation.EQ)
        sen_pre_22.addAddend(1, self.inputVars_verified[1][16])
        sen_pre_22.setScalar(0)
        disjunction_pre_31.append([sen_pre_21])
        disjunction_pre_32.append([sen_pre_22])
        self.network_verified.addDisjunctionConstraint(disjunction_pre_31)
        self.network_verified.addDisjunctionConstraint(disjunction_pre_32)

        #####################
        ### Post-condition ##
        #####################
        # Leverage a|(b&c) = (a|b)&(a|c)
        # (N1 = 0 and N2 = 1) or (N1 = 1 and N2 = 0)
        # It becomes (N1 = 0 or N2 = 0) and (N1 = 1 or N2 = 1)
        disjunction_post_1 = []
        diff_post_1 = MarabouCore.Equation(MarabouCore.Equation.GE)
        diff_post_1.addAddend(1, self.outputVars_verified[0][0])
        diff_post_1.addAddend(-1, self.outputVars_verified[0][1])
        diff_post_1.setScalar(
            0
        )  # XXX implementation problem, only support 0 (easy to find CE.) or 0.01 (might be hard to find CE.)
        diff_post_2 = MarabouCore.Equation(MarabouCore.Equation.GE)
        diff_post_2.addAddend(1, self.outputVars_verified[1][0])
        diff_post_2.addAddend(-1, self.outputVars_verified[1][1])
        diff_post_2.setScalar(0)
        disjunction_post_1.append([diff_post_1])
        disjunction_post_1.append([diff_post_2])

        disjunction_post_2 = []
        diff_post_3 = MarabouCore.Equation(MarabouCore.Equation.GE)
        diff_post_3.addAddend(1, self.outputVars_verified[0][1])
        diff_post_3.addAddend(-1, self.outputVars_verified[0][0])
        diff_post_3.setScalar(0)
        diff_post_4 = MarabouCore.Equation(MarabouCore.Equation.GE)
        diff_post_4.addAddend(1, self.outputVars_verified[1][1])
        diff_post_4.addAddend(-1, self.outputVars_verified[1][0])
        diff_post_4.setScalar(0)
        disjunction_post_2.append([diff_post_3])
        disjunction_post_2.append([diff_post_4])

        self.network_verified.addDisjunctionConstraint(disjunction_post_1)
        self.network_verified.addDisjunctionConstraint(disjunction_post_2)

        vals, stats = self.network_verified.solve()

        return vals
