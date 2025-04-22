import gamelib
import random
import math
import warnings
from sys import maxsize
import json

class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()
        seed = random.randrange(maxsize)
        random.seed(seed)
        # gamelib.debug_write('Random seed: {}'.format(seed))

        self.firstLineWall = [[23,12],[22,11],[21,10],[20,9]]
        for x in range(4):
            self.firstLineWall.append([x,13])
            self.firstLineWall.append([27 - x,13])

        for x in range(8,20):
            self.firstLineWall.append([x,8])

        self.firstLineWall.append([1,12])
        self.firstLineWall.append([4,9])


        self.moreWalls = [[8,9],[7,10],[6,11],[5,12]]
        self.turret_locations = [[5,11],[6,10],[7,9]] # [23,12],[22,11],[21,10],[20,9]
        self.turret_locations.append([4,11])
        self.turret_locations.append([4,9])
        # self.turret_locations += [[25,11],[24,10],[23,9]]
        self.support_locations = [[8,7], [19,7]]

        # add more moreSupportLocations and buiild them when SP > 10
        self.moreSupportLocations = []
        for x in range(8,20):
            self.moreSupportLocations.append([x, 7])
        
        self.portToOpen = []
        self.portOpened = False

        self.right_kamikaze_attack_locations = [[24, 14], [25, 14], [26, 14], [27, 14], [24, 15], [25, 15], [26, 15], [24,16], [25,16]]
        self.left_kamikaze_attack_locations = [[0, 14], [1, 14], [2, 14], [3, 14], [1, 15], [2, 15], [3, 15], [2,16], [3,16]]

        self.will_kamikaze_attack = False
        self.kamikaze_attack_location = 0

    def on_game_start(self, config):
        """ 
        Read in config and perform any initial setup here 
        """
        # gamelib.debug_write('Configuring your custom algo strategy...')
        self.config = config
        global WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR, MP, SP, LEFT_KAMIKAZE, RIGHT_KAMIKAZE
        WALL = config["unitInformation"][0]["shorthand"]
        SUPPORT = config["unitInformation"][1]["shorthand"]
        TURRET = config["unitInformation"][2]["shorthand"]
        SCOUT = config["unitInformation"][3]["shorthand"]
        DEMOLISHER = config["unitInformation"][4]["shorthand"]
        INTERCEPTOR = config["unitInformation"][5]["shorthand"]
        MP = 1
        SP = 0
        # This is a good place to do initial setup
        LEFT_KAMIKAZE = 1
        RIGHT_KAMIKAZE = 2
        self.scored_on_locations = []

    def on_turn(self, turn_state):
        """
        This function is called every turn with the game state wrapper as
        an argument. The wrapper stores the state of the arena and has methods
        for querying its state, allocating your current resources as planned
        unit deployments, and transmitting your intended deployments to the
        game engine.
        """
        game_state = gamelib.GameState(self.config, turn_state)
        # gamelib.debug_write('Performing turn {} of your custom algo strategy'.format(game_state.turn_number))
        game_state.suppress_warnings(True)  #Comment or remove this line to enable warnings.

        if game_state.turn_number < 1:
            self.early_game(game_state)
        else:
            self.mid_game(game_state)

        game_state.submit_turn()


    """
    NOTE: All the methods after this point are part of the sample starter-algo
    strategy and can safely be replaced for your custom algo.
    """
    def early_game(self,game_state):
        # game_state.attempt_spawn(SCOUT, [14, 0], 5)
        self.build_wall(game_state)
        self.build_tower(game_state)
        self.build_support(game_state)
        self.attack(game_state)

    def mid_game(self, game_state):
        """
        For defense we will use a spread out layout and some interceptors early on.
        We will place turrets near locations the opponent managed to score on.
        For offense we will use long range demolishers if they place stationary units near the enemy's front.
        If there are no stationary units to attack in the front, we will send Scouts to try and score quickly.
        """
        if not self.firstLineWallIsFull(game_state):
            self.will_kamikaze_attack = False
            self.kamikaze_attack_location = 0

        if self.will_kamikaze_attack:
            if self.kamikaze_attack_location == LEFT_KAMIKAZE:
                game_state.attempt_spawn(WALL, [4,12])
            self.kamikaze_attack(game_state)
            self.will_kamikaze_attack = False
            self.kamikaze_attack_location = 0
        else:
            if game_state.game_map[4,12] and game_state.game_map[4,12][0].unit_type == WALL:
                game_state.attempt_remove([4,12])

            self.check_potential_enemy_attack(game_state)
            self.build_tower(game_state)
            self.build_wall(game_state)
            self.build_support(game_state)

            self.replaceWall(game_state)

            self.update_tower(game_state)
            self.update_support(game_state)
            self.update_wall(game_state)
            self.spawnInterceptor(game_state)
        
        if game_state.get_resource(MP) > 10 and self.firstLineWallIsFull(game_state):
            self.will_kamikaze_attack = True
            self.kamikaze_attack_location = self.plan_kamikaze_attack(game_state)

        game_state.attempt_spawn(WALL, self.moreWalls)
        game_state.attempt_upgrade(self.moreWalls)
        
        self.attack(game_state)

        # Lastly, if we have spare SP, let's build some supports
        # maybe use support earlier instead of upgrading walls.
        # dynamic move (e.g. if i got damaged on specif location, more defender at that point)
        # dynamic move (check the health of specific terittory and predict if scout can do damage)
        #               also check what happen if enemy place new towers.

        for x, y in self.moreSupportLocations:
            game_state.attempt_spawn(SUPPORT, [x,y])
            game_state.attempt_upgrade([x,y])
    
    def firstLineWallIsFull(self, game_state):
        walls = [[x,8] for x in range(8,20)] + [[20,9],[21,10],[22,11],[23,12],[24,13]]
        for [x,y] in walls:
            if len(game_state.game_map[x,y]) == 0:
                return False
        return True

    def build_tower(self, game_state):
        game_state.attempt_spawn(TURRET, self.turret_locations)

    def build_support(self, game_state):
        # Build supports in positions that shield our mobile units
        game_state.attempt_spawn(SUPPORT, self.support_locations)

    def build_wall(self, game_state):
        if self.portOpened:
            for [x,y] in self.firstLineWall:
                if [x,y] not in self.portToOpen:  
                    game_state.attempt_spawn(WALL, [x,y])
            # reset
            self.portOpened = False
            self.portOpened = []
        else:
            game_state.attempt_spawn(WALL, self.firstLineWall)

        # if top right wall are low hp, build tower at edge (potentially, enemy will attack to this side)
        if self.top_right_wall_weak(game_state):
            game_state.attempt_spawn(TURRET, [[26,12],[25,11],[22,10]])

    def top_right_wall_weak(self, game_state):
        top_right_wall = [[24,13],[23,12],[22,11]]
        for [x,y] in top_right_wall:
            if len(game_state.game_map[x, y]) == 0:
                continue
            unit = game_state.game_map[x, y][0]
            currHP = unit.health
            originHP = unit.max_health
            if currHP / originHP < 0.6:
                return True
            
        return False

    def update_wall(self, game_state):
        game_state.attempt_upgrade(self.firstLineWall)

    def update_support(self, game_state):
        game_state.attempt_upgrade(self.support_locations)
    
    def update_tower(self, game_state):
        game_state.attempt_upgrade(self.turret_locations)

    def spawnInterceptor(self, game_state):
        enemyMP = game_state.get_resource(MP,1)
        # gamelib.debug_write('enemyMP {}'.format(enemyMP))
        base = 6.0

        enemy_support_counter = min(self.count_enemy_support(game_state),3)

        if enemyMP >= base * 3:
            game_state.attempt_spawn(INTERCEPTOR, [7, 6], 2 + int(enemy_support_counter / 3))
            game_state.attempt_spawn(INTERCEPTOR, [23, 9], 2 + int(enemy_support_counter / 3))
        elif enemyMP >= base * 2:
            # game_state.attempt_spawn(INTERCEPTOR, [8, 5], 2)
            game_state.attempt_spawn(INTERCEPTOR, [7, 6], 1 + int(enemy_support_counter / 3))
            game_state.attempt_spawn(INTERCEPTOR, [23, 9], 1 + int(enemy_support_counter / 3))
        elif enemyMP >= base:
            game_state.attempt_spawn(INTERCEPTOR, [7, 6], 1 + int(enemy_support_counter / 3))
            game_state.attempt_spawn(INTERCEPTOR, [23, 9], 1 + int(enemy_support_counter / 3))
        elif enemy_support_counter > 0:
            game_state.attempt_spawn(INTERCEPTOR, [7, 6], 1 + int(enemy_support_counter / 3))
            game_state.attempt_spawn(INTERCEPTOR, [23, 9], int(enemy_support_counter / 3))

        if self.top_right_wall_weak(game_state):
                game_state.attempt_spawn(INTERCEPTOR, [23, 9], int(enemy_support_counter / 3))

    def replaceWall(self, game_state):    
        allWalls = self.moreWalls + self.firstLineWall
        for [x,y] in allWalls:
            tile = game_state.game_map[x, y]
            if len(tile) > 0:
                tile = tile[0]
                if tile.unit_type != WALL:
                    continue
            else:
                continue
            currHealth = tile.health 
            originHealth = tile.max_health 
            if currHealth/originHealth < 0.5:
                game_state.attempt_remove([x,y])
                dirs = [
                        [1,0],
                        [0,1],
                        [0,-1],
                        [1,1],
                        [1,-1],
                        ]
                # for dir in dirs:
                #     if [x + dir[0], y + dir[1]] == [5,10]:
                #         continue
                #     game_state.attempt_spawn(WALL, [x + dir[0], y + dir[1]])

    # or check enemy structure point, if less than 5, send
    def attack(self,game_state):
        # Sending more at once is better since attacks can only hit a single scout at a time
        # check if demolisher needed
        enemyTowerCount = 0.0
        enemyWallCount = 0.0
        for y in range(14,20):
            for x in range(0,20-y):
                x = 5 - x
                if len(game_state.game_map[x,y]) > 0 and game_state.game_map[x,y][0].unit_type == TURRET:
                    enemyTowerCount += 1.0
                    if game_state.game_map[x,y][0].upgraded:
                        enemyTowerCount += 0.5

                if len(game_state.game_map[x,y]) > 0 and game_state.game_map[x,y][0].unit_type == WALL:
                    enemyWallCount += 1.0
                    if game_state.game_map[x,y][0].upgraded:
                        enemyWallCount += 0.5

        numberOfDemolisher = int(enemyTowerCount + enemyWallCount // 4) 
        # wait one round to generate more MP
        if numberOfDemolisher <= 5 and numberOfDemolisher * 3 > (game_state.get_resource(MP)):
            return
        
        # if enemyTowerCount > 0:# and self.portOpened:  
            # game_state.attempt_spawn(DEMOLISHER, self.portToOpen, numberOfDemolisher)

        if enemyTowerCount == 0 and enemyWallCount > 0 and game_state.get_resource(MP) >= 9.0:
            game_state.attempt_spawn(DEMOLISHER, [14, 0], 1)

        path = game_state.find_path_to_edge([14,0])
        final_position = path[-1]
        if self.ideal_exit(game_state, final_position) and self.is_path_clear_of_enemy_walls(game_state,path) and game_state.get_resource(MP) >= 5:
            game_state.attempt_spawn(SCOUT, [14, 0], 1000)

    def ideal_exit(self, game_state, final_position):
        """Check if the left-side diagonal path [0,14] to [13,27] is clear (no wall or turret)."""
        
        for i in range(6):
            x = i
            y = 14 + i
            if [x,y] == final_position:
                return True
        return False

    def is_path_clear_of_enemy_walls(self, game_state, path):
        for loc in path:
            # gamelib.debug_write("path: {}".format(loc))
            
            units = game_state.game_map[loc[0], loc[1]]
            for unit in units:
                if unit.player_index == 1 and unit.unit_type in [WALL, TURRET]:
                    return False
        return True

    def count_enemy_support(self, game_state):
        counter = 0
        for i in range(14):
            y = 14 + i
            for x in range(i,28 - i):
                if len(game_state.game_map[x,y]) > 0 and game_state.game_map[x,y][0].unit_type == SUPPORT:
                    counter += 1

        return counter

    def check_potential_enemy_attack(self, game_state):
        if (len(game_state.game_map[1,14]) > 0 and game_state.game_map[1,14][0].pending_removal) or \
            (len(game_state.game_map[2,14]) > 0 and game_state.game_map[2,14][0].pending_removal):
            game_state.attempt_remove([[1,13],[2,13]])
            self.portToOpen.append([1,13])
            self.portToOpen.append([2,13])
            self.portOpened = True

        if (len(game_state.game_map[25,14]) > 0 and game_state.game_map[25,14][0].pending_removal) or \
            (len(game_state.game_map[26,14]) > 0 and game_state.game_map[26,14][0].pending_removal):
            game_state.attempt_remove([[25,13],[26,13]])
            self.portToOpen.append([25,13])
            self.portToOpen.append([26,13])
            self.portOpened = True

    def plan_kamikaze_attack(self, game_state) -> int:
        # check if enemy has a turret on the top right corner
        attackRight = True
        # determine if there are weak points on the right side
        rightScore = 0
        leftScore = 0
        for [x,y] in self.right_kamikaze_attack_locations:
            if game_state.contains_stationary_unit([x,y]):
                rightScore += 1
        for [x,y] in self.left_kamikaze_attack_locations:
            if game_state.contains_stationary_unit([x,y]):
                leftScore += 1

        if rightScore < leftScore:
            attackRight = True
        else:
            attackRight = False

        if attackRight:
            # sell wall on the [26,13]
            if game_state.game_map[25,13] and game_state.game_map[25,13][0].unit_type == WALL:
                game_state.attempt_remove([[25,13],[26,13]])
            return RIGHT_KAMIKAZE
        else:
            # sell wall on [2, 13]
            if game_state.game_map[2,13] and game_state.game_map[2,13][0].unit_type == WALL:
                game_state.attempt_remove([[1,12],[1,13]])
            # sell wall on [3,10]
            if game_state.game_map[4,9] and game_state.game_map[4,9][0].unit_type == WALL:
                game_state.attempt_remove([4,9])
            return LEFT_KAMIKAZE

    def kamikaze_attack(self, game_state):
        if self.kamikaze_attack_location == RIGHT_KAMIKAZE:
            game_state.attempt_spawn(INTERCEPTOR, [24,10], 4)
            game_state.attempt_spawn(SCOUT, [13,0], game_state.number_affordable(SCOUT))
        else:
            game_state.attempt_spawn(INTERCEPTOR, [3,10], 4)
            game_state.attempt_spawn(SCOUT, [14,0], game_state.number_affordable(SCOUT))

        if game_state.game_map[4,12] and game_state.game_map[4,12][0].unit_type == WALL:
                game_state.attempt_remove([4,12])
        
   

if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()