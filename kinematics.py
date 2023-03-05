import numpy as np
import SwarmPackagePy


class IRB120:
    """
    This class is a simple implementation of IRB6700 robot our implementation contains forward and inverse kinematic
    functions we used Numpy for mathematical and array based calculations and SwarmPackagePy for GreyWolf algorithm.
    Numpy documentation :
    https://numpy.org/doc/

    Gray Wolf Algorithm
    which is a mimics the leadership hierarchy and hunting mechanism of gray wolves in nature. Wolves live in a pack.
     The average pack consists of a family of 5–12 animals. wolves have strict social hierarchy which is represented
     by the division of a pack into four levels: alpha, beta, delta, and omega.
     full doc can be found here in developers github address :
     https://github.com/SISDevelop/SwarmPackagePy#gray-wolf-optimization
    """
    def __init__(self, axis_limitations=((-165, 165), (-110, 110), (-110, 70), (-160, 160), (-120, 120), (-400, 400)),
                 end_effector_position=(0, 0, 0), upper_bound=5, lower_bound=-5):

        self.axis_limitations = list(axis_limitations)
        self.axis_current_angles = list(np.zeros(6))
        self.end_effector_position = list(end_effector_position)
        self.end_effector_goal = list(np.zeros(3))
        self.upper_scale_bound = upper_bound
        self.lower_scale_bound = lower_bound

    def scale_bound_setter(self, upper_bound: int, lower_bound: int) -> None:
        """This is just a simple setter for our upper and lower scale boundaries.
        we transform our GA problem into that scale in order to find the answers faster than original range

        Args:
            upper_bound(int): upper bound of our limit
            lower_bound(int): lower bound of our limit

        Returns:
                None
        """
        self.upper_scale_bound = upper_bound
        self.lower_scale_bound = lower_bound

    def end_effector_setter(self, new_coordination: list) -> None:
        """Just a setter for self.end_effector_goal

        Args:
            new_coordination(list): new values to update end effector position

        Returns:
                None
        """
        self.end_effector_position = new_coordination

    def axis_current_angle_setter(self, new_coordination: list) -> None:
        """Just a setter for self.axis_current_angle

        Args:
            new_coordination(list): new values to update

        Returns:
                None
        """
        self.axis_current_angles = new_coordination

    def distance_calculator(self, coordinate1: list, coordinate2: list) -> float:
        """Simple mathematical distance calculation function

        Args:
            coordinate1(list): first coordination
            coordinate2(list): second coordination

        Returns:
            float: distance of two given coordination
        """
        distances = list()

        for a, b in zip(coordinate1, coordinate2):
            distances.append((a - b) ** 2)

        final_distance = sum(distances) ** 0.5

        return  final_distance

    def rotation_calculator(self, coordinate1: list, coordinate2: list) -> list:
        """calculates the rotation from point 1 to point 2

        Args:
            coordinate1(list): first coordinates
            coordinate2(list): second coordination

        Returns:
            list : calculated rotations
        """
        rotation = list()
        for a, b in zip(coordinate1, coordinate2):
            rotation.append(a-b)
        return rotation

    def create_dh_matrix(self, theta: float, alpha: float, d: float, r: float) -> np.ndarray:
        """A function to make and return the DH matrix based on given arguments

            Args:
                theta(float): theta in degrees
                alpha(float): alpha in degrees
                d(float): mm
                r(float): in mm

            Returns:
                np.ndarray: a 4x4 matrix  known as DH matrix
            """

        # Creating our initializer matrix ( all zeros at first )
        dh_matrix = np.zeros((4, 4), dtype=float)
        # theta and alpha should be converted to radians for np.cos() and np.sin() function
        theta_c = np.cos(np.radians(theta))
        theta_s = np.sin(np.radians(theta))
        alpha_c = np.cos(np.radians(alpha))
        alpha_s = np.sin(np.radians(alpha))

        # initializing our matrix
        dh_matrix[0, 0] = theta_c
        dh_matrix[0, 1] = -1 * theta_s * alpha_c
        dh_matrix[0, 2] = theta_s * alpha_s
        dh_matrix[0, 3] = r * theta_c

        dh_matrix[1, 0] = theta_s
        dh_matrix[1, 1] = theta_c * alpha_c
        dh_matrix[1, 2] = -1 * theta_c * alpha_s
        dh_matrix[1, 3] = r * theta_s

        dh_matrix[2, 1] = alpha_s
        dh_matrix[2, 2] = alpha_c
        dh_matrix[2, 3] = d

        dh_matrix[3, 3] = 1

        return dh_matrix

    def forward_kinematics(self, thetas: list) -> list:
        """

            Args:
                thetas(list) : list of thetas for each link (theta[0]  to theta[5] )

            Returns:
                list: coordination to the point that end effector goes
            """
        dh_for_link0 = self.create_dh_matrix(theta=thetas[0], alpha=0, r=0, d=0)
        dh_for_link1 = self.create_dh_matrix(theta=thetas[1], alpha=0, r=0, d=0.290)
        dh_for_link2 = self.create_dh_matrix(theta=thetas[2], alpha=0, r=0, d=0.270)
        dh_for_link3 = self.create_dh_matrix(theta=thetas[3], alpha=0, r=0, d=0.070)
        dh_for_link4 = self.create_dh_matrix(theta=thetas[4], alpha=0, r=0.302, d=0)
        dh_for_link5 = self.create_dh_matrix(theta=thetas[5], alpha=0, r=0.72, d=0)

        final_coordination = dh_for_link0 @ dh_for_link1 @ dh_for_link2 @ dh_for_link3 @ dh_for_link4 @ dh_for_link5 @ \
            np.array([0, 0, 0, 1])

        return final_coordination[0:3]

    def validate_degrees(self, thetas: list) -> tuple:
        """
            Checking input thetas with motion of our Axis

            Args:
                thetas(list): list of thetas
            Returns:
                tuple : A boolean to show validation and contains a message as String
            """
        # Checking number of thetas
        if len(thetas) > 6:
            return False, 'Too many entries , only 6 is required'
        elif len(thetas) < 6:
            return False, 'Not enough entries , 6 is required'

        # Validation operation for all of our thetas
        if not (-165 <= thetas[0] <= 165):
            return False, 'Check Axis 1, the valid range is between  -170 and 170'
        if not (-110 <= thetas[1] <= 110):
            return False, 'Check Axis 2, , the valid range is between -180 and 70'
        if not (-110 <= thetas[2] <= 70):
            return False, 'Check Axis 3, the valid range is between -180 and 70 '
        if not (-160 <= thetas[3] <= 160):
            return False, 'Check Axis 4, the valid range is between -300 and 300'
        if not (-120 <= thetas[4] <= 120):
            return False, 'Check Axis 5, the valid range is between -130 and 130'
        if not (-400 <= thetas[5] <= 400):
            return False, 'Check Axis 6, the valid range is between -360 and 360'

        return True, 'Fine'

    def scaled_to_unscaled_degrees(self, scaled_axis: list) -> list:
        """Genetic algorithm that we use gives us scaled data between a lower and higher limit for each axis so we have
        to de-scale it to our axis limitations

        Args:
            scaled_axis(list): scaled degrees that genetic algorithm (Grey Wolf) give us

        Returns:
            list : list of calculated degrees

        """

        unscaled_axis = list()
        for i in range(0, len(self.axis_limitations)):
            unscaled_axis.append(((scaled_axis[i] / (self.upper_scale_bound - self.lower_scale_bound)) *
                        (self.axis_limitations[i][1] - self.axis_limitations[i][0])) + self.axis_limitations[i][0])

        return unscaled_axis

    def custom_cost_function(self, thetas_for_cost_test: list) -> float:
        """We need cost function instead of fitness function, because default of the algorithm that we use goes for
        minimized answers and label them as better ones , so the lower the cost the better answer .

        Args:
            thetas_for_cost_test(list): list of scaled thetas that GreyWolf algorithm gives us
        Returns:
                float: the calculated cost based on distance formula
        """

        unscaled_axis = self.scaled_to_unscaled_degrees(thetas_for_cost_test)
        cost = self.distance_calculator(self.end_effector_goal, self.forward_kinematics(unscaled_axis))

        # Checking the validation of the founded answer
        if self.validate_degrees(unscaled_axis)[0]:
            return cost
        else:
            # We have to sue invalid answers
            return cost + 1000

    def inverse_kinematics(self, coordinates: list, lower_bound=None, upper_bound=None) -> tuple:
        """Applying inverse kinematics using a GA algorithm to optimize our answers

        Args:
            coordinates(list): the goal of of our end effector
            upper_bound(int): upper bound of our scale(optional)
            lower_bound(int): lower bound of our scale (optional)
        Returns:
                tuple: contains rotations and distance between the goal and our answer
        """
        # Setting our given goal
        self.end_effector_goal = coordinates

        # if user gives us custom boundaries we change them
        if lower_bound is not None and upper_bound is not None:
            self.scale_bound_setter(upper_bound, lower_bound)

        # GreyWolf algorithm parameters based on library documentation
        # n : number of population that we create every iteration
        # function : our cost function , could be from library but we implemented a custom one
        # lb , ub = lower and upper bound of our scale
        # dimension = our problem dimension ( number of answers in a simple way)
        # iteration :  number of algorithm iterations
        grey_wolf_optimizer = SwarmPackagePy.gwo(n=30, function=self.custom_cost_function, lb=self.lower_scale_bound,
            ub=self.upper_scale_bound, dimension=len(self.axis_limitations), iteration=600)

        # Gives us wolf pack leaders, we just need Alpha ( the wolf pack leader )
        optimized_data = grey_wolf_optimizer.get_leaders()[0]
        optimized_data = self.scaled_to_unscaled_degrees(optimized_data)

        answer = (self.distance_calculator(self.forward_kinematics(optimized_data), self.end_effector_position),
        self.rotation_calculator(optimized_data, self.axis_current_angles))

        return answer


if __name__ == '__main__':
    # Simple initializer and menu just for " testing purposes "
    my_robot = IRB120()

    while True:
        user_choice = int(input('Choose one of the options \n'
                                '1- forward kinematics \n'
                                '2- Inverse kinematics \n'
                                '3- Exit \n'))
        if user_choice == 1:
            thetas = input("Please enter thetas (6 thetas , separated with space) \n ").split()
            thetas = list(map(float, thetas))
            try:
                if not my_robot.validate_degrees(thetas)[0]:
                    raise ValueError
                print(my_robot.forward_kinematics(thetas))

            except ValueError:
                print(my_robot.validate_degrees(thetas)[1])
        elif user_choice == 2:
            flag = 0
            try:
                coordination = input('Please enter X , Y , Z , separated with space \n').split()
                coordination = list(map(float, coordination))

                if len(coordination) < 3:
                    flag = -1
                    raise ValueError
                elif len(coordination) > 3:
                    flag = 1
                    raise ValueError

                user_answer = my_robot.inverse_kinematics(coordination)
                print("Distance : \n  {0} \n Rotations for [X, Y, Z] : \n {1}".format(user_answer[0], user_answer[1]))

            except ValueError:
                if flag == -1:
                    print("We need at least 3 arguments")
                elif flag == 1:
                    print("Too many arguments , only 3 is required")

        elif user_choice == 3:
            break
        else:
            print('Wrong option , Only 1, 2 and 3 is allowed')
