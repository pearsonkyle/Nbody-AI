# to run in background: 
# nohup python -u generate_simulations.py > log.txt &

import pickle 

from nbody.simulation import randomize, generate, integrate, analyze, report

if __name__ == "__main__":

    Nsamples = 10000

    for ii in range(Nsamples):

        try:
            print('simulation:',ii)

            objects = randomize()
            sim = generate(objects)
            sim_data = integrate(sim, objects, 365, 365*24) # year long integrations, timestep = 1 hour 
            ttv_data = analyze(sim_data)
            pickle.dump(ttv_data,open('simulations/ttv_{}.pkl'.format(ii),'wb') )
            report(ttv_data, savefile='simulations/report_{}.png'.format(ii))
        except:
            pass