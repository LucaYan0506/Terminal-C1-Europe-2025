import gamelib
import random
import math
import warnings
from sys import maxsize
import json


"""
Most of the algo code you write will be in this file unless you create new
modules yourself. Start by modifying the 'on_turn' function.

Advanced strategy tips: 

  - You can analyze action frames by modifying on_action_frame function

  - The GameState.map object can be manually manipulated to create hypothetical 
  board states. Though, we recommended making a copy of the map to preserve 
  the actual current map state.
"""

class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()
        seed = random.randrange(maxsize)
        random.seed(seed)
        # gamelib.debug_write('Random seed: {}'.format(seed))
        self.portOpened = False
        
        self.firstLineWall = [[8,11],[7,12],[6,13],[5,13],[8,10],[9,10],[10,9],[24,10],[25,11],[26,13],[27,13],[23,9]]
        self.firstLineWall.append([0,13])
        self.firstLineWall.append([1,13])
        self.firstLineWall.append([2,13])
        for x in range(12,21):
            self.firstLineWall.append([x,8])

        self.firstLineWall.append([22,8])
        # self.firstLineWall.append([21,9])
        self.firstLineWall.append([11,9])
        self.firstLineWall.append([21,8])
     

        # self.moreWalls = []
        self.moreWalls = [[23,10],[25, 13], [25, 12], [24, 11],[24,12],[22,8]]
        self.turret_locations = [[6,12],[5,10],[4,11],[3,12],[26,12]]
        

        self.support_locations = [[5,8]]
        
        self.portToOpen = [2,11]
    def on_game_start(self, config):
        """ 
        Read in config and perform any initial setup here 
        """
        # gamelib.debug_write('Configuring your custom algo strategy...')
        self.config = config
        global WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR, MP, SP
        WALL = config["unitInformation"][0]["shorthand"]
        SUPPORT = config["unitInformation"][1]["shorthand"]
        TURRET = config["unitInformation"][2]["shorthand"]
        SCOUT = config["unitInformation"][3]["shorthand"]
        DEMOLISHER = config["unitInformation"][4]["shorthand"]
        INTERCEPTOR = config["unitInformation"][5]["shorthand"]
        MP = 1
        SP = 0
        # This is a good place to do initial setup
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
        game_state.attempt_spawn(SCOUT, [14, 0], 5)
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
        self.build_tower(game_state)
        self.build_support(game_state)
        self.build_wall(game_state)
        game_state.attempt_spawn(WALL, self.moreWalls)
        game_state.attempt_upgrade(self.moreWalls)

        self.replaceWall(game_state)

        self.update_tower(game_state)
        self.update_support(game_state)
        self.update_wall(game_state)
        self.spawnInterceptor(game_state)
       
        self.attack(game_state)

        if game_state.get_resource(MP) > 15:
            affordable = game_state.number_affordable(SCOUT)
            if affordable > 0:
                game_state.attempt_spawn(SCOUT, [3, 10], affordable)
                # gamelib.debug_write("Spawned {} scouts at [3,10] with {} MP".format(affordable, game_state.get_resource(MP)))

        # Lastly, if we have spare SP, let's build some supports
        # maybe use support earlier instead of upgrading walls.
        # dynamic move (e.g. if i got damaged on specif location, more defender at that point)
        # dynamic move (check the health of specific terittory and predict if scout can do damage)
        #               also check what happen if enemy place new towers.

    def build_tower(self, game_state):
        game_state.attempt_spawn(TURRET, self.turret_locations)

    def build_support(self, game_state):
        # Build supports in positions that shield our mobile units
        game_state.attempt_spawn(SUPPORT, self.support_locations)

    def build_wall(self, game_state):
        if self.portOpened:
            for [x,y] in self.firstLineWall:
                if [x,y] != self.portToOpen:  
                    game_state.attempt_spawn(WALL, [x,y])
        else:
            game_state.attempt_spawn(WALL, self.firstLineWall)

    def update_wall(self, game_state):
        game_state.attempt_upgrade(self.firstLineWall)

    def update_support(self, game_state):
        game_state.attempt_upgrade(self.support_locations)
    
    def update_tower(self, game_state):
        game_state.attempt_upgrade(self.turret_locations)

    def spawnInterceptor(self, game_state):
        enemyMP = game_state.get_resource(MP,1)
        # gamelib.debug_write('enemyMP {}'.format(enemyMP))
        base = 8.0
        if game_state.turn_number < 5:
            base = 6.0
        if enemyMP >= base * 3:
            # game_state.attempt_spawn(INTERCEPTOR, [8, 5], 2)
            game_state.attempt_spawn(INTERCEPTOR, [2, 11], 4)
        elif enemyMP >= base * 2:
            # game_state.attempt_spawn(INTERCEPTOR, [8, 5], 2)
            game_state.attempt_spawn(INTERCEPTOR, [2, 11], 2)
        elif enemyMP >= base:
            game_state.attempt_spawn(INTERCEPTOR, [2, 11], 1)

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

    def attack(self,game_state):
        # Sending more at once is better since attacks can only hit a single scout at a time
        # check if demolisher needed
        enemyTowerCount = 0
        there_is_wall = False
        for y in range(14,19):
            for x in range(0,19-y):
                # gamelib.debug_write('game_state.game_map[4-x,y] {}'.format(game_state.game_map[4-x,y]))
                if len(game_state.game_map[4-x,y]) > 0 and game_state.game_map[4-x,y][0].unit_type == TURRET:
                    enemyTowerCount+=1
                if len(game_state.game_map[4-x,y]) > 0 and game_state.game_map[4-x,y][0].unit_type == WALL:
                    there_is_wall = True

        
        # wait one round to generate more MP
        if not self.portOpened and enemyTowerCount > (game_state.get_resource(MP)) // int(3*1.5):
            return
        
        # if enemyTowerCount > 0 and not self.portOpened:                       
        #     game_state.attempt_remove(self.portToOpen)
        #     # gamelib.debug_write('open port')
        #     self.portOpened = True
        #     return
        if enemyTowerCount > 0:# and self.portOpened:  
            game_state.attempt_spawn(DEMOLISHER, self.portToOpen, int(enemyTowerCount*1.5))
            # self.portOpened = False

        if enemyTowerCount == 0 and there_is_wall and game_state.get_resource(MP) >= 9.0:
            game_state.attempt_spawn(DEMOLISHER, [14, 0], 1)

        path = game_state.find_path_to_edge([14,0])
        final_position = path[-1]
        if self.ideal_exit(game_state, final_position) and self.is_path_clear_of_enemy_walls(game_state,path):
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




    def stall_with_interceptors(self, game_state):
        """
        Send out interceptors at random locations to defend our base from enemy moving units.
        """
        # We can spawn moving units on our edges so a list of all our edge locations
        friendly_edges = game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_LEFT) + game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_RIGHT)
        
        # Remove locations that are blocked by our own structures 
        # since we can't deploy units there.
        deploy_locations = self.filter_blocked_locations(friendly_edges, game_state)
        
        # While we have remaining MP to spend lets send out interceptors randomly.
        while game_state.get_resource(MP) >= game_state.type_cost(INTERCEPTOR)[MP] and len(deploy_locations) > 0:
            # Choose a random deploy location.
            deploy_index = random.randint(0, len(deploy_locations) - 1)
            deploy_location = deploy_locations[deploy_index]
            
            game_state.attempt_spawn(INTERCEPTOR, deploy_location)
            """
            We don't have to remove the location since multiple mobile 
            units can occupy the same space.
            """

    def demolisher_line_strategy(self, game_state):
        """
        Build a line of the cheapest stationary unit so our demolisher can attack from long range.
        """
        # First let's figure out the cheapest unit
        # We could just check the game rules, but this demonstrates how to use the GameUnit class
        stationary_units = [WALL, TURRET, SUPPORT]
        cheapest_unit = WALL
        for unit in stationary_units:
            unit_class = gamelib.GameUnit(unit, game_state.config)
            if unit_class.cost[game_state.MP] < gamelib.GameUnit(cheapest_unit, game_state.config).cost[game_state.MP]:
                cheapest_unit = unit

        # Now let's build out a line of stationary units. This will prevent our demolisher from running into the enemy base.
        # Instead they will stay at the perfect distance to attack the front two rows of the enemy base.
        for x in range(27, 5, -1):
            game_state.attempt_spawn(cheapest_unit, [x, 11])

        # Now spawn demolishers next to the line
        # By asking attempt_spawn to spawn 1000 units, it will essentially spawn as many as we have resources for
        game_state.attempt_spawn(DEMOLISHER, [24, 10], 1000)

    def least_damage_spawn_location(self, game_state, location_options):
        """
        This function will help us guess which location is the safest to spawn moving units from.
        It gets the path the unit will take then checks locations on that path to 
        estimate the path's damage risk.
        """
        damages = []
        # Get the damage estimate each path will take
        for location in location_options:
            path = game_state.find_path_to_edge(location)
            damage = 0
            for path_location in path:
                # Get number of enemy turrets that can attack each location and multiply by turret damage
                damage += len(game_state.get_attackers(path_location, 0)) * gamelib.GameUnit(TURRET, game_state.config).damage_i
            damages.append(damage)
        
        # Now just return the location that takes the least damage
        return location_options[damages.index(min(damages))]

    def detect_enemy_unit(self, game_state, unit_type=None, valid_x = None, valid_y = None):
        total_units = 0
        for location in game_state.game_map:
            if game_state.contains_stationary_unit(location):
                for unit in game_state.game_map[location]:
                    if unit.player_index == 1 and (unit_type is None or unit.unit_type == unit_type) and (valid_x is None or location[0] in valid_x) and (valid_y is None or location[1] in valid_y):
                        total_units += 1
        return total_units
        
    def filter_blocked_locations(self, locations, game_state):
        filtered = []
        for location in locations:
            if not game_state.contains_stationary_unit(location):
                filtered.append(location)
        return filtered

    def on_action_frame(self, turn_string):
        """
        This is the action frame of the game. This function could be called 
        hundreds of times per turn and could slow the algo down so avoid putting slow code here.
        Processing the action frames is complicated so we only suggest it if you have time and experience.
        Full doc on format of a game frame at in json-docs.html in the root of the Starterkit.
        """
        # Let's record at what position we get scored on
        state = json.loads(turn_string)
        events = state["events"]
        breaches = events["breach"]
        for breach in breaches:
            location = breach[0]
            unit_owner_self = True if breach[4] == 1 else False
            # When parsing the frame data directly, 
            # 1 is integer for yourself, 2 is opponent (StarterKit code uses 0, 1 as player_index instead)
            if not unit_owner_self:
                # gamelib.debug_write("Got scored on at: {}".format(location))
                self.scored_on_locations.append(location)
                # gamelib.debug_write("All locations: {}".format(self.scored_on_locations))


if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()
