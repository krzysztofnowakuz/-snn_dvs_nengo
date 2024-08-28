import nengo
import numpy as np
import matplotlib.pyplot as plt

# Definiujemy model
with nengo.Network() as model:
    # Tworzymy źródło danych - funkcję sinusoidalną
    input_node = nengo.Node(output=np.sin)
    
    # Tworzymy warstwę neuronów typu LIF (leaky integrate-and-fire)
    ensemble = nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.LIF())
    
    # Tworzymy połączenie między źródłem danych a neuronami
    nengo.Connection(input_node, ensemble)
    
    # Tworzymy probe do śledzenia aktywności neuronów
    probe = nengo.Probe(ensemble.neurons)

# Uruchamiamy symulację
with nengo.Simulator(model) as sim:
    sim.run(1.0)  # Symulujemy przez 1 sekundę

# Wyświetlamy wyniki
plt.figure()
plt.plot(sim.trange(), sim.data[probe])
plt.xlabel('Czas (s)')
plt.ylabel('Aktywność neuronów')
plt.show()
