package NEATLib;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class NEAT{
    // IDEA: Make User changeable
    /// Maximum delta between specimen
    double MAX_DELTA = 2.0;
    /// Importance of excess genes for the delta
    double WEIGHT_C1 = 1.0;
    /// Importance of disjoint genes for the delta
    double WEIGHT_C2 = 1.0;
    /** Importance of average weight difference between matching genes for the 
     * delta */
    double WEIGHT_C3 = 1.0;
    /** Propability of a change in structure as opposed to a change in 
     * connections */
    double PROP_STRUCTURE = 0.5;
    /// Propability of creating a new connection as opposed to splitting one
    double PROP_CONNECTION = 0.5;
    /// Propability of toggling a connection instead of adjusting weight
    double PROP_TOGGLE = 0.5;
    /// Percentage of a species to be killed before mating and mutating
    double KILL_OFF_PORTION = 0.5;
    
    /// List of all specimen in a generation
    List<NEATNetwork> nets = new ArrayList<>();
    /// Global tracker of innovations made by the networks
    List<IntegerPair> innovations = new ArrayList<>();
    
    /// Amount of networks in any given generation
    int networkCount;
    /// Amount of input nodes in a network
    int inputCount;
    /// Amount of output values in a network
    int outputCount;
    
    // *STRUCTORS --------------------------------------------------------------
    
    // IDEA: Add a seed for randomisation
    // IDEA: Implement BIAS
    /**
     * Constructor for a new NEAT run
     * @param inNodes Amount of input nodes
     * @param outNodes Amount of output nodes
     * @param networks Amount of simultaneous networks
     * @param bias decide wether or not to use a bias
     */
    public NEAT(int inNodes, int outNodes,int networks, boolean bias)
    {
        networkCount = networks;
        inputCount = inNodes;
        outputCount = outNodes;
        for(int i = 0; i < networks; i++){
            nets.add(new NEATNetwork(inNodes,outNodes,this));
        }
    }
    
    // METHODS -----------------------------------------------------------------
    
    /**
     * Add the fitness to a specified net
     * @param value Fitness value to be added
     * @param index index of the network in the "nets"-list
     */ 
    public void addFitness(double value, int index)
    {
        nets.get(index).addFitness(value);
    }
    
    // IDEA: Hide from users
    /**
     * Tries to add a new innovation. If the innovation already existed return
     * index + 1 as innovation number. Else return the new innovation number.
     * @param inNodeIndex Index of the input node.
     * @param outNodeIndex Index of the output node.
     * @return Returns the appropriate innovations.
     */
    public int addInnovation(int inNodeIndex, int outNodeIndex)
    {
        IntegerPair newInnovation = new IntegerPair(inNodeIndex,outNodeIndex);
        
        for(IntegerPair ip : innovations)
        {
            if (ip.compare(newInnovation))
            {
                return innovations.indexOf(ip) + 1;
            }
        }
        
        innovations.add(newInnovation);
        return innovations.size();
    }
    
    /**
     * Advances the nets into the next generation
     */
    public void advanceGeneration()
    {
        // Sort list for fitness
        Collections.sort(nets, (NEATNetwork a, NEATNetwork b) -> {
            return a.fitness > b.fitness ? 1 : (a.fitness < b.fitness ? -1 : 0);
        });
        
        // Generate species and assign nets to them
        List<List<NEATNetwork>> speciesList = new ArrayList<>();
        
        // Get the list of all specimen
        for(NEATNetwork net : nets)
        {
            // Check every species for affiliation
            boolean hasFoundSpecies = false;
            for(List<NEATNetwork> species : speciesList)
            {
                // If the network fits a species add it to it.
                if(species.get(0).compareCompatibility(net, WEIGHT_C1, WEIGHT_C2, WEIGHT_C3) < MAX_DELTA)
                {
                    species.add(net);
                    hasFoundSpecies = true;
                    break;
                }
            }

            // If no species has been found establish a new one.
            if(!hasFoundSpecies)
            {
                List<NEATNetwork> newSpecies = new ArrayList<>();
                newSpecies.add(net);
                speciesList.add(newSpecies);
            }
        }
        
        // Clear the network list
        nets.clear();
        
        // Share the fitness in the species
        double absoluteFitness = 0.0;
        
        for (List<NEATNetwork> species : speciesList)
        {
            for(NEATNetwork net : species)
            {
                net.fitness /= (double) species.size();
                absoluteFitness += net.fitness;
            }
        }
        
        // Remove up to the @KILL_OF_PORTION worst of the species and
        // figure out how to generate the new specimen
        for(List<NEATNetwork> species : speciesList)
        {
            // Get the species portion of the fitness
            double speciesFitness = 0.0;
            for(NEATNetwork specimen : species)
            {
                speciesFitness += specimen.fitness;
            }
            double fitnessPortion = speciesFitness / absoluteFitness;
            
            // Kill off the @KILL_OFF_PORTION of a species except if the species
            // only consists of one specimen
            if(species.size() == 1)
            {
                species.get(0).mutate();
                nets.add(species.get(0));
                if(nets.size() == networkCount) {break;}
                continue;
            }
            
            int addCount = (int) (networkCount * fitnessPortion);
            if(addCount + nets.size() > networkCount)
            {
                addCount += nets.size() - networkCount;
            }
            
            double specimenPortion = 1.0 / species.size();
            for(double d = 0.0; d < KILL_OFF_PORTION; d += specimenPortion)
            {
                species.remove(species.size() - 1);
            }
            
            // Add offspring from every species up to the wanted counted 
            int added = 0;
            for(int i = 0; i < species.size() - 1; i++)
            {
                for(int j = i+1; j < species.size(); j++)
                {
                    nets.add(species.get(0).mate(species.get(i),species.get(j)));
                    added++;
                    if(added == addCount){break;}
                }
                if(added == addCount){break;}
            }
            if(added == addCount){continue;}
            
            // Fill up remaining spots with mutations
            for(int i = 0; i < species.size(); i++)
            {
                species.get(i).mutate();
                nets.add(species.get(i));
                added++;
                if(added == addCount){break;}
            }
        }
        
        // Should 100 nets not have been reached add new basic networks
        for(int i = nets.size(); i < networkCount; i++)
        {
            nets.add(new NEATNetwork(inputCount, outputCount, this));
        }
    }
    
    /**
     * Print the current maximum fitness
     * @param high Highest previously achieved value
     */
    public void printMaxFitness()
    {
        double maxFitness = Double.NEGATIVE_INFINITY;
        
        for(NEATNetwork net : nets)
        {
            if( net.fitness > maxFitness)
            {
                maxFitness = net.fitness;
            }
        }
        
        System.out.println(maxFitness);
    }
    
    /**
     * Tells a specific network to process the given inputs and return the 
     * result.
     * @param inputs Array of double values to be used in the input nodes.
     * @param index Index of the network in the "nets"-list
     * @throws IllegalArgumentException If the input array is not congruent to
     * the inputs an IllegalArgumentException is thrown.
     * @return Returns an array the output value. 
     */
    public double[] processNetwork(double[] inputs,int index)
    {
        if(inputs.length != inputCount){throw new IllegalArgumentException();}
        
        return nets.get(index).process(inputs);
    }
    
    /**
     * Sets the fitness of all specimen back to 0;
     */
    public void resetFitness()
    {
        for(NEATNetwork specimen : nets)
        {
            specimen.fitness = 0.0;
        }
    }
    
    // INTERNAL CLASSES --------------------------------------------------------
    
    /**
     * Supporting class for a List entry. Very basic implementation of an
     * integer pair.
     */
    class IntegerPair
    {
        /// First integer of the pair
        public int first;
        /// Second integer of the pair
        public int second;
        
        /**
         * Construct a new integer pair
         * @param first The first value
         * @param second The second value
         */
        public IntegerPair(int first, int second)
        {
            this.first = first;
            this.second = second;
        }
        
        /**
         * Compare two integer pairs
         * @param p The pair to be compared to
         * @return Returns wether the two integer pairs are equal
         */
        public boolean compare(IntegerPair p)
        {
            return (this.first == p.first && this.second == p.second);
        }
    }
}