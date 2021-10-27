import time
import flappy_bird_gym
import neat
import os
env = flappy_bird_gym.make("FlappyBird-v0")


def eval_genomes(genomes, config):

    for _, genome in genomes:
        observation = env.reset()  
        done = False
        net = neat.nn.FeedForwardNetwork.create(genome, config) #create NN for each genome
        genome.fitness = 0  #Starting fitness
        while not done:
            output = net.activate(observation)  #Getting output from net based on observations
            action = 1 if output[0]>0.5 else 0
        
            observation, reward, done, info = env.step(action)  
            genome.fitness += reward    #Rewards genome for each turn
        env.reset()

def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    

    # Run for up to 100 generations.
    winner = p.run(eval_genomes, 100)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

    #render the best model
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    render_model(winner_net)

def render_model(winner):
    
    observation = env.reset()
 
    done = False
   
  
    while not done:
        time.sleep(1 / 30)
        env.render()   #Render game      
        output = winner.activate(observation)
       
        action = 1 if output[0]>0.5 else 0
        observation, reward, done, info = env.step(action)
    env.reset()

if __name__ == '__main__':
    #path to config file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)

    env.close()