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
        gamelib.debug_write('Random seed: {}'.format(seed))

    def on_game_start(self, config):
        """ 
        Read in config and perform any initial setup here 
        """
        gamelib.debug_write('Configuring your custom algo strategy...')
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
        self.attack_side = "left"  # Set a default attack side rather than None
        # Attempt to place a funnel pattern
        self.funnel_opening_location = [13, 11]  # Default gap in our defenses
        self.early_game_threshold = 5   # Even more reduced for faster attacks
        self.attack_threshold = 10      # Attack much earlier
        
        # Adaptive defense tracking
        self.current_health = 30
        self.health_history = []
        self.weak_points = set()
        self.last_breach_turn = -1
        self.consecutive_breach_turns = 0
        self.attack_mode = "balanced"  # Can be "balanced", "aggressive", or "defensive"
        
        # Offensive tracking
        self.last_attack_turn = -1
        self.attack_success = False
        self.attack_cooldown = 0
        self.default_mp_reserve = 2  # Reduced reserve for more attacks

    def on_turn(self, turn_state):
        """
        This function is called every turn with the game state wrapper as
        an argument. The wrapper stores the state of the arena and has methods
        for querying its state, allocating your current resources as planned
        unit deployments, and transmitting your intended deployments to the
        game engine.
        """
        game_state = gamelib.GameState(self.config, turn_state)
        gamelib.debug_write('Performing turn {} of your custom algo strategy'.format(game_state.turn_number))
        game_state.suppress_warnings(True)  #Comment or remove this line to enable warnings.

        # Track health changes to detect breaches
        self.track_health_changes(game_state)
        
        # Update attack side periodically based on enemy defense
        if game_state.turn_number % 3 == 0 or self.attack_side is None:
            self.update_attack_side(game_state)

        # Implement strategy
        self.adaptive_strategy(game_state)

        game_state.submit_turn()

    def track_health_changes(self, game_state):
        """
        Track health changes to detect when we're being scored on.
        Adjust strategy based on health losses.
        """
        current_health = game_state.my_health
        self.health_history.append(current_health)
        
        # We need at least 2 turns of health data to detect changes
        if len(self.health_history) < 2:
            return
            
        # Check if we lost health from the previous turn
        health_change = current_health - self.health_history[-2]
        
        if health_change < 0:
            # We lost health! Adjust strategy based on severity
            self.last_breach_turn = game_state.turn_number
            self.consecutive_breach_turns += 1
            
            # Only go defensive if we're being severely breached
            if self.consecutive_breach_turns >= 3:  # Increased threshold from 2 to 3
                self.attack_mode = "defensive"
                gamelib.debug_write(f"DEFENSIVE MODE ACTIVATED - Lost health {-health_change} in consecutive turns!")
            else:
                # Single breach, stay balanced but reinforce
                self.attack_mode = "balanced"
                gamelib.debug_write(f"Health reduced by {-health_change}. Reinforcing defenses.")
        else:
            # Reset consecutive breach counter if we didn't lose health
            self.consecutive_breach_turns = 0
            
            # If we haven't lost health in a while, go aggressive
            turns_since_last_breach = game_state.turn_number - self.last_breach_turn
            if turns_since_last_breach > 5 and game_state.turn_number > self.early_game_threshold:  # Reduced from 10 to 5
                self.attack_mode = "aggressive"
                gamelib.debug_write("AGGRESSIVE MODE ACTIVATED - No breaches detected recently!")
            elif turns_since_last_breach > 3:  # Reduced from 5 to 3
                self.attack_mode = "balanced"

    def adaptive_strategy(self, game_state):
        """
        Enhanced strategy that adapts based on health loss patterns and enemy moves.
        More aggressive with attacks while maintaining adaptive defense.
        """
        # Always prioritize defense repair if we recently lost health
        if game_state.turn_number == self.last_breach_turn + 1:
            self.repair_breaches(game_state)
        
        # Build base defenses regardless of mode
        self.build_enhanced_defenses(game_state)
        
        # Build reactive defense against scored locations
        self.build_reactive_defense(game_state)
        
        # Always attack every turn after early game
        # Maintain defensive structures but focus on consistent attack pressure
        if game_state.turn_number < self.early_game_threshold:
            # Early game setup - build basic defenses and start probing
            self.build_support_grid(game_state)
            self.upgrade_key_defenses(game_state)  # Only upgrade key defenses in early game
            
            # Early aggressive probing with scouts starting from turn 1
            if game_state.get_resource(MP) >= 4:  # Reduced threshold from 6 to 4
                gamelib.debug_write("Early game scout rush")
                self.scout_rush(game_state, aggressive=True)
        else:
            # Always maintain supports for shielding
            self.build_support_grid(game_state)
            
            # Execute attack strategy every turn based on current mode
            if self.attack_mode == "aggressive":
                # Extremely aggressive attacks with minimal defense upgrades
                self.execute_aggressive_attacks(game_state)
                
                # Only upgrade defenses if we have excess SP
                if game_state.get_resource(SP) >= 15:
                    self.upgrade_key_defenses(game_state)
                    
            elif self.attack_mode == "defensive":
                # Defensive mode - still attack but more balanced
                self.upgrade_defenses(game_state)
                self.execute_defensive_attacks(game_state)
                
            else:
                # Balanced mode - good attacks with reasonable defense
                self.upgrade_key_defenses(game_state)
                self.execute_balanced_attacks(game_state)

    def repair_breaches(self, game_state):
        """
        Analyzes and repairs breaches in our defense based on recent scored_on_locations.
        """
        # Focus on the most recent breach locations
        recent_breaches = self.scored_on_locations[-3:] if len(self.scored_on_locations) > 3 else self.scored_on_locations
        
        if not recent_breaches:
            return
            
        gamelib.debug_write(f"Repairing breaches at: {recent_breaches}")
        
        for location in recent_breaches:
            # Build a mini-fortress at each breach point
            
            # Place a wall at the breach location to block future units
            if game_state.can_spawn(WALL, location):
                game_state.attempt_spawn(WALL, location)
                game_state.attempt_upgrade([location])
                
            # Add a turret behind the wall
            turret_loc = [location[0], location[1]+1]
            if game_state.can_spawn(TURRET, turret_loc):
                game_state.attempt_spawn(TURRET, turret_loc)
                game_state.attempt_upgrade([turret_loc])
                
            # Add walls on the sides if possible
            side_locs = [
                [location[0]-1, location[1]],
                [location[0]+1, location[1]]
            ]
            
            for side_loc in side_locs:
                if game_state.can_spawn(WALL, side_loc):
                    game_state.attempt_spawn(WALL, side_loc)

    def execute_aggressive_attacks(self, game_state):
        """
        Execute aggressive attacks with minimal resource reservation.
        Increased attack frequency and unit counts.
        """
        # Always attempt to attack if attack cooldown is 0
        enemy_front_line_strength = self.detect_enemy_unit(game_state, unit_type=None, valid_x=None, valid_y=[14, 15])
        
        # Debug print resources
        gamelib.debug_write(f"MP: {game_state.get_resource(MP)}, SP: {game_state.get_resource(SP)}")
        
        # Check MP and decide on attack type - even lower threshold
        if game_state.get_resource(MP) >= 5:  # Reduced from 7 to 5
            # If enemy has strong front, use demolishers
            if enemy_front_line_strength > 6 and game_state.get_resource(MP) >= 10:  # Reduced from 12 to 10
                gamelib.debug_write("Executing demolisher attack")
                self.demolisher_attack_strategy(game_state, aggressive=True)
            else:
                # Otherwise go with scouts for faster scoring
                gamelib.debug_write("Executing scout rush")
                self.scout_rush(game_state, aggressive=True)
                
            # Add a second wave of units if we have lots of MP
            if game_state.get_resource(MP) >= 5:
                # If we still have resources, send more units from the other side
                old_attack_side = self.attack_side
                self.attack_side = "right" if old_attack_side == "left" else "left"
                
                # Send a smaller second wave
                if game_state.get_resource(MP) >= 8:
                    gamelib.debug_write("Sending second wave")
                    self.scout_rush(game_state, aggressive=False)
                    
                # Restore original attack side
                self.attack_side = old_attack_side
        else:
            gamelib.debug_write(f"Not enough MP to attack: {game_state.get_resource(MP)}")

    def execute_defensive_attacks(self, game_state):
        """
        Even in defensive mode, maintain some offensive pressure.
        """
        # Still send at least some scouts in defensive mode
        if game_state.turn_number % 2 == 0 and game_state.get_resource(MP) >= 7:
            gamelib.debug_write("Executing defensive scout attack")
            self.scout_rush(game_state, aggressive=False)
            
        # Send interceptors to defend
        if game_state.get_resource(MP) >= 3:
            gamelib.debug_write("Sending defensive interceptors")
            self.send_interceptors(game_state, 1)

    def execute_balanced_attacks(self, game_state):
        """
        Balanced offense and defense with more regular attacks.
        """
        enemy_front_line_strength = self.detect_enemy_unit(game_state, unit_type=None, valid_x=None, valid_y=[14, 15])
        
        # Debug print resources
        gamelib.debug_write(f"Balanced - MP: {game_state.get_resource(MP)}, SP: {game_state.get_resource(SP)}")
        
        # Attack on every turn if possible
        if game_state.get_resource(MP) >= 6:  # Reduced from 8 to 6
            if enemy_front_line_strength > 8:
                # Enemy has strong front line defenses, use demolishers
                gamelib.debug_write("Balanced demolisher attack")
                self.demolisher_attack_strategy(game_state, aggressive=False)
            else:
                # Enemy has weaker front line, use scouts
                gamelib.debug_write("Balanced scout attack")
                self.scout_rush(game_state, aggressive=False)
                
            # Add interceptors on alternating turns if we have enough MP
            if game_state.turn_number % 2 == 0 and game_state.get_resource(MP) >= 3:
                gamelib.debug_write("Sending balanced interceptors")
                self.send_interceptors(game_state, 1)
        else:
            gamelib.debug_write(f"Not enough MP for balanced attack: {game_state.get_resource(MP)}")

    def upgrade_key_defenses(self, game_state):
        """
        Only upgrade the most important defensive structures to save SP for building more.
        """
        # Upgrade key funnel walls
        key_walls = [[8, 11], [9, 11], [17, 11], [18, 11]]
        game_state.attempt_upgrade(key_walls)
        
        # Upgrade key turrets that cover the funnel
        key_turrets = [[12, 10], [15, 10]]
        game_state.attempt_upgrade(key_turrets)
        
        # Upgrade key supports for shielding
        key_supports = [[13, 5], [14, 5]]
        game_state.attempt_upgrade(key_supports)

    def scout_rush(self, game_state, aggressive=False):
        """
        Send a rush of scouts, with aggression level determining resource allocation.
        Increased unit counts for more offensive pressure.
        """
        # Ensure attack_side is set
        if self.attack_side is None:
            self.attack_side = "left"  # Default to left if not set
            
        if self.attack_side == "left":
            spawn_location = [6, 5]
        else:
            spawn_location = [21, 5]
            
        # Calculate how many scouts to send based on aggression
        scout_cost = game_state.type_cost(SCOUT)[MP]
        
        if aggressive:
            # In aggressive mode, keep minimal MP in reserve
            reserve_mp = self.default_mp_reserve
        else:
            # In normal mode, keep slightly more in reserve
            reserve_mp = self.default_mp_reserve + 2
            
        available_mp = max(0, game_state.get_resource(MP) - reserve_mp)
        scout_count = int(available_mp // scout_cost)
        
        gamelib.debug_write(f"Scout rush - available MP: {available_mp}, scout cost: {scout_cost}, count: {scout_count}, location: {spawn_location}")
        
        # Make sure spawn location is not blocked
        if not game_state.can_spawn(SCOUT, spawn_location):
            # Try alternative spawn points
            friendly_edges = game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_LEFT) + game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_RIGHT)
            deploy_locations = self.filter_blocked_locations(friendly_edges, game_state)
            
            if not deploy_locations:
                gamelib.debug_write("No valid spawn locations for scouts!")
                return
                
            if self.attack_side == "left":
                possible_locations = [loc for loc in deploy_locations if loc[0] < 14]
            else:
                possible_locations = [loc for loc in deploy_locations if loc[0] >= 14]
                
            if possible_locations:
                spawn_location = possible_locations[0]
                gamelib.debug_write(f"Using alternative scout spawn: {spawn_location}")
            else:
                spawn_location = deploy_locations[0]
                gamelib.debug_write(f"Using fallback scout spawn: {spawn_location}")
        
        if scout_count > 0:
            result = game_state.attempt_spawn(SCOUT, spawn_location, scout_count)
            gamelib.debug_write(f"Scout spawn result: {result} units at {spawn_location}")
            self.last_attack_turn = game_state.turn_number

    def send_interceptors(self, game_state, count=1):
        """
        Send interceptors to defend or support attacks.
        """
        # Ensure attack_side is set
        if self.attack_side is None:
            self.attack_side = "left"  # Default to left if not set
            
        friendly_edges = game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_LEFT) + game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_RIGHT)
        deploy_locations = self.filter_blocked_locations(friendly_edges, game_state)
        
        if not deploy_locations:
            gamelib.debug_write("No valid spawn locations for interceptors!")
            return
            
        # Choose locations based on attack side
        if self.attack_side == "left":
            interceptor_locations = [loc for loc in deploy_locations if loc[0] > 14]
        else:
            interceptor_locations = [loc for loc in deploy_locations if loc[0] < 14]
            
        # Default to any available location if filtering eliminated all options
        if not interceptor_locations:
            interceptor_locations = deploy_locations
            
        # Spawn interceptors up to count or until MP runs out
        spawned = 0
        for i in range(min(count, len(interceptor_locations))):
            if game_state.get_resource(MP) >= game_state.type_cost(INTERCEPTOR)[MP]:
                result = game_state.attempt_spawn(INTERCEPTOR, interceptor_locations[i])
                if result > 0:
                    spawned += 1
                    gamelib.debug_write(f"Interceptor spawned at {interceptor_locations[i]}")
                
        if spawned == 0 and count > 0:
            gamelib.debug_write("Failed to spawn any interceptors!")

    def upgrade_defenses(self, game_state):
        """
        Upgrade key defensive structures to make them more resilient.
        """
        # Upgrade all walls in the funnel
        funnel_walls = [[8, 11], [9, 11], [10, 11], [11, 11], [12, 11], 
                        [14, 11], [15, 11], [16, 11], [17, 11], [18, 11]]
        game_state.attempt_upgrade(funnel_walls)
        
        # Upgrade corner walls for added protection
        corner_walls = [[0, 13], [1, 13], [26, 13], [27, 13]]
        game_state.attempt_upgrade(corner_walls)
        
        # Upgrade turrets
        turret_locations = [[3, 11], [5, 9], [8, 6], [12, 10], [15, 10], [19, 6], [22, 9], [24, 11]]
        additional_turrets = [[11, 9], [16, 9], [13, 8], [14, 8]]
        
        # Prioritize turrets in different ways based on mode
        if self.attack_mode == "defensive":
            # Upgrade all turrets in defensive mode if possible
            game_state.attempt_upgrade(turret_locations + additional_turrets)
        else:
            # Otherwise just upgrade key turrets
            key_turret_upgrades = [[12, 10], [15, 10], [11, 9], [16, 9]]
            game_state.attempt_upgrade(key_turret_upgrades)

    def build_enhanced_defenses(self, game_state):
        """
        Build enhanced defenses taking advantage of the modified unit stats.
        Focuses on a funnel design with adaptive reinforcement based on breach history.
        Optimized to use fewer resources while maintaining effectiveness.
        """
        # First, build a strong front line of walls
        wall_locations = [[0, 13], [1, 13], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9], [7, 8], [8, 7], 
                          [19, 7], [20, 8], [21, 9], [22, 10], [23, 11], [24, 12], [25, 13], [26, 13], [27, 13]]
        
        # Add a funnel in the middle to direct enemy units
        funnel_walls = [[8, 11], [9, 11], [10, 11], [11, 11], [12, 11], 
                         [14, 11], [15, 11], [16, 11], [17, 11], [18, 11]]
        
        # Add second line of defense behind the funnel - use fewer walls to save SP
        second_line = [[10, 10], [13, 9], [14, 9], [17, 10]]  # Reduced from 8 to 4 walls
        
        # Place walls, prioritizing funnel walls first, then corners, then selective second line
        for location in funnel_walls:
            if game_state.can_spawn(WALL, location):
                game_state.attempt_spawn(WALL, location)
                
        for location in wall_locations:
            if game_state.can_spawn(WALL, location):
                game_state.attempt_spawn(WALL, location)
                
        for location in second_line:
            if game_state.can_spawn(WALL, location):
                game_state.attempt_spawn(WALL, location)
        
        # Place essential turrets - reduced count to save SP
        turret_locations = [[3, 11], [8, 6], [12, 10], [15, 10], [19, 6], [24, 11]]  # Removed 2 turrets
        
        # Add only the most critical second line turrets
        additional_turrets = [[11, 9], [16, 9]]  # Reduced from 4 to 2 turrets
        
        # In defensive mode, add more turrets, otherwise keep minimal
        if self.attack_mode == "defensive":
            all_turrets = turret_locations + additional_turrets
        else:
            # In aggressive mode, build even fewer turrets
            if self.attack_mode == "aggressive":
                all_turrets = [[12, 10], [15, 10], [3, 11], [24, 11]]  # Only 4 key turrets
            else:
                all_turrets = turret_locations
            
        for location in all_turrets:
            if game_state.can_spawn(TURRET, location):
                game_state.attempt_spawn(TURRET, location)
        
        # Upgrade key funnel walls
        key_wall_upgrades = [[8, 11], [9, 11], [17, 11], [18, 11]]
        game_state.attempt_upgrade(key_wall_upgrades)

    def build_support_grid(self, game_state):
        """
        Build support structures to take advantage of the increased shield range and health.
        Position and quantity of supports varies by mode.
        Optimized for minimal SP usage.
        """
        # Core supports that should always be built
        support_locations = [[13, 5], [14, 5]]  # Reduced from 4 to 2 core supports
        
        # Additional supports for better coverage
        additional_supports = [[10, 7], [17, 7]]  # Reduced from 4 to 2 additional supports
        
        # Build supports based on mode
        if self.attack_mode == "aggressive":
            target_supports = support_locations
        elif self.attack_mode == "defensive":
            target_supports = support_locations + additional_supports + [[9, 4], [18, 4]]
        else:
            target_supports = support_locations + additional_supports
            
        for location in target_supports:
            if game_state.can_spawn(SUPPORT, location):
                game_state.attempt_spawn(SUPPORT, location)
        
        # Upgrade key supports
        key_support_upgrades = [[13, 5], [14, 5]]
        game_state.attempt_upgrade(key_support_upgrades)
        
        # Only upgrade additional supports in defensive mode
        if self.attack_mode == "defensive" and game_state.get_resource(SP) > 8:
            additional_upgrades = [[10, 7], [17, 7]]
            game_state.attempt_upgrade(additional_upgrades)

    def build_reactive_defense(self, game_state):
        """
        This function builds reactive defenses based on where the enemy scored on us from.
        """
        for location in self.scored_on_locations:
            # Build turret one space above so that it doesn't block our own edge spawn locations
            build_location = [location[0], location[1]+1]
            # Place a wall in front to protect the turret
            wall_location = [location[0], location[1]+2]
            
            if game_state.can_spawn(TURRET, build_location):
                game_state.attempt_spawn(TURRET, build_location)
                # If we built a turret, consider upgrading it since turrets are less powerful now
                game_state.attempt_upgrade(build_location)
            
            if game_state.can_spawn(WALL, wall_location):
                game_state.attempt_spawn(WALL, wall_location)
                game_state.attempt_upgrade(wall_location)

    def update_attack_side(self, game_state):
        """
        Analyze enemy defenses to decide which side to focus attacks on.
        """
        left_defenses = self.detect_enemy_unit(game_state, unit_type=None, valid_x=list(range(0, 14)), valid_y=[14, 15, 16, 17])
        right_defenses = self.detect_enemy_unit(game_state, unit_type=None, valid_x=list(range(14, 28)), valid_y=[14, 15, 16, 17])
        
        # Attack the side with fewer defenses
        if left_defenses <= right_defenses:
            self.attack_side = "left"
        else:
            self.attack_side = "right"

    def stall_with_interceptors(self, game_state):
        """
        Send out interceptors at random locations to defend our base from enemy mobile units.
        """
        # Send interceptors based on current mode
        if self.attack_mode == "defensive":
            count = 2  # More interceptors in defensive mode
        elif self.attack_mode == "aggressive":
            count = 0  # None in aggressive mode
        else:
            count = 1  # Balanced mode
            
        if count > 0:
            gamelib.debug_write(f"Stalling with {count} interceptors")
            self.send_interceptors(game_state, count)

    def demolisher_attack_strategy(self, game_state, aggressive=False):
        """
        Build a line of cheap stationary units so our demolisher can attack from long range.
        Increased demolisher counts for stronger attacks.
        """
        # Ensure attack_side is set
        if self.attack_side is None:
            self.attack_side = "left"  # Default to left if not set
            
        cheapest_unit = WALL
        
        # Build a line based on which side we're attacking
        if self.attack_side == "left":
            deploy_location = [8, 5]
            wall_line = [[x, 10] for x in range(9, 14)]
        else:
            deploy_location = [19, 5]
            wall_line = [[x, 10] for x in range(14, 19)]
        
        # Check if deploy location is valid
        if not game_state.can_spawn(DEMOLISHER, deploy_location):
            # Find alternative location
            friendly_edges = game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_LEFT) + game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_RIGHT)
            deploy_locations = self.filter_blocked_locations(friendly_edges, game_state)
            
            if not deploy_locations:
                gamelib.debug_write("No valid spawn locations for demolishers!")
                return
                
            if self.attack_side == "left":
                possible_locations = [loc for loc in deploy_locations if loc[0] < 14]
            else:
                possible_locations = [loc for loc in deploy_locations if loc[0] >= 14]
                
            if possible_locations:
                deploy_location = possible_locations[0]
                gamelib.debug_write(f"Using alternative demolisher spawn: {deploy_location}")
            else:
                deploy_location = deploy_locations[0]
                gamelib.debug_write(f"Using fallback demolisher spawn: {deploy_location}")
            
        # Build walls if resources allow - use fewer walls to save SP
        # Only build every other wall to save resources while still providing a path
        for i, location in enumerate(wall_line):
            if i % 2 == 0:  # Only build every other wall
                game_state.attempt_spawn(cheapest_unit, location)
        
        # Determine demolisher count based on aggression - increased counts
        if aggressive:
            demo_count = 6 if game_state.get_resource(MP) >= 18 else 4  # Increased from 4 to 6
        else:
            demo_count = 3 if game_state.get_resource(MP) >= 10 else 2  # Increased from 2 to 3
        
        gamelib.debug_write(f"Demolisher attack - MP: {game_state.get_resource(MP)}, count: {demo_count}, location: {deploy_location}")
            
        result = game_state.attempt_spawn(DEMOLISHER, deploy_location, demo_count)
        gamelib.debug_write(f"Demolisher spawn result: {result} units at {deploy_location}")
        self.last_attack_turn = game_state.turn_number
        
        # Add scouts if we have excess MP
        if game_state.get_resource(MP) >= 5:  # Reduced from 6 to 5
            scout_location = [deploy_location[0], deploy_location[1]-1]
            # Check if this location is valid
            if not game_state.can_spawn(SCOUT, scout_location):
                # Find the closest valid point
                for dx in range(-2, 3):
                    for dy in range(-2, 0):
                        test_loc = [deploy_location[0] + dx, deploy_location[1] + dy]
                        if game_state.can_spawn(SCOUT, test_loc):
                            scout_location = test_loc
                            break
            
            scout_count = 4 if aggressive else 3  # Increased from 3/2 to 4/3
            result = game_state.attempt_spawn(SCOUT, scout_location, scout_count)
            gamelib.debug_write(f"Support scout spawn result: {result} units at {scout_location}")

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
                # Note: Turrets now deal 4 damage (was 5) and upgraded turrets deal 8 damage (was 16)
                enemies = game_state.get_attackers(path_location, 0)
                for enemy in enemies:
                    if enemy.unit_type == TURRET:
                        if enemy.upgraded:
                            damage += 8
                        else:
                            damage += 4
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
                gamelib.debug_write("Got scored on at: {}".format(location))
                self.scored_on_locations.append(location)
                gamelib.debug_write("All locations: {}".format(self.scored_on_locations))


if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()
