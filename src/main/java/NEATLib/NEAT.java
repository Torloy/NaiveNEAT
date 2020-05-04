package NEATLib;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class NEAT{
	//// Speciation PARAMETERS
    // Maximum delta between specimen
    double MAX_DELTA = 3;
    // Importance of excess genes for the delta
    double WEIGHT_C1 = 1.0;
    // Importance of disjoint genes for the delta
    double WEIGHT_C2 = 1.0;
    /** Importance of average weight difference between matching genes for the 
     * delta */
    double WEIGHT_C3 = .4;
    
    //// GENERATION ADVANCEMENT PARAMETERS
    // Speciation threshold
    int CHAMPION_THRESHOLD = 5;
    // Portion of a species to be added as a mutation
    double PORTION_MUTATION = .25;
    
    //// MUTATION PARAMETERS
    // Maximum perturbance of a weight
    double MAX_PERTURBANCE = 8.0;
    // Probability of adding a new connection
    double PROP_CONNECTION = .5;
    // Probability of carrying over the disabled epigene
    double PROP_KEEP_DISABLED = .75;
    // Probability of adding a new node
    double PROP_NODE = .5;
    // Probability of mutating the weights
    double PROP_WEIGHT = .8;
    // Probability of getting uniform weight perturbance
    double PROP_WEIGHT_UNIFORM = .9;
    // Size of the range of a random weight
    double RANDOM_WEIGHT_RANGE = 5.0;
    
    //// PROCESSING PARAMETER
    // The Sigmoid modifier to tune the sigmoid activation
    double SIGMOID_MODIFIER = 4.9;
    
    
    // List of all specimen in a generation
    List<NEATNetwork> nets = new ArrayList<>();
    // Global tracker of innovations made by the networks
    List<IntegerPair> innovations = new ArrayList<>();
    
    // Amount of networks in any given generation
    int networkCount;
    // Amount of input nodes in a network
    int inputCount;
    // Amount of output values in a network
    int outputCount;
    // Whether the BIAS is used
    boolean usesBias = false;
    
    // *STRUCTORS --------------------------------------------------------------
    
    /**
     * Constructor for a new NEAT run
     * @param inNodes Amount of input nodes
     * @param outNodes Amount of output nodes
     * @param networks Amount of simultaneous networks
     * @param isBiasEnabled decide whether or not to use a bias
     */
    public NEAT(int inNodes, int outNodes,int networks, boolean isBiasEnabled)
    {
        networkCount = networks;
        inputCount = inNodes;
        outputCount = outNodes;
        usesBias = isBiasEnabled;
        
        for(int i = 0; i < networks; i++){
            nets.add(new NEATNetwork(inNodes + (isBiasEnabled ? 1 : 0),outNodes,this));
        }
    }
    
    // METHODS -----------------------------------------------------------------
    
    // TODO: Allow to extract network
    
    /**
     * Add the fitness to a specified net
     * @param value Fitness value to be added
     * @param index index of the network in the "nets"-list
     */ 
    public void addFitness(double value, int index)
    {
        nets.get(index).addFitness(value);
    }
    
    /**
     * Tries to add a new innovation. If the innovation already existed return
     * index + 1 as innovation number. Else return the new innovation number.
     * @param inNodeIndex Index of the input node.
     * @param outNodeIndex Index of the output node.
     * @return Returns the appropriate innovations.
     */
    public int addInnovation(int inNodeIndex, int outNodeIndex)
    {
    	if(inNodeIndex == -1 || outNodeIndex == -1) {
    		System.out.print("");
    	}
    	
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
            return a.fitness > b.fitness ? -1 : (a.fitness < b.fitness ? 1 : 0);
        });
        

        // Generate species and assign nets to them
        List<List<NEATNetwork>> speciesList = new ArrayList<>();
        
        // Get the list of all specimen
        for(NEATNetwork specimen : nets)
        {
            // Check every species for affiliation
            boolean hasFoundSpecies = false;
            for(List<NEATNetwork> species : speciesList)
            {
                // If the network fits a species add it to it.
                if(species.get(0).compareCompatibility(specimen, WEIGHT_C1, WEIGHT_C2, WEIGHT_C3) < MAX_DELTA)
                {
                    species.add(specimen);
                    hasFoundSpecies = true;
                    break;
                }
            }

            // If no species has been found establish a new one.
            if(!hasFoundSpecies)
            {
                List<NEATNetwork> newSpecies = new ArrayList<>();
                newSpecies.add(specimen);
                speciesList.add(newSpecies);
            }
        }
        
        // Clear the network list
        nets.clear();
        
        // Calculate the amount for each species
        Map<Integer,Double> fitnesses = new TreeMap<Integer,Double>();
        fitnesses.put(-1, 0.0);
        for(int i = 0; i < speciesList.size(); i++) {
        	fitnesses.put(i, 0.0);
        	
        	for(NEATNetwork specimen : speciesList.get(i)) {
        		fitnesses.put(i, fitnesses.get(i) + specimen.fitness);
        	}
        	
        	fitnesses.put(i, fitnesses.get(i) / speciesList.get(i).size());
        	fitnesses.put(-1, fitnesses.get(i) + fitnesses.get(-1));
        }
        
        for(int i = 0; i < speciesList.size(); i++) {
        	fitnesses.put(i, (fitnesses.get(i) * networkCount) / fitnesses.get(-1));
        }
        
        Map<Integer,Long> amounts = new TreeMap<Integer,Long>();
        for(int i = 0; i < speciesList.size(); i++) {
        	amounts.put(i, Math.round(fitnesses.get(i)));
        	double adjusment = (fitnesses.get(i) - Math.round(fitnesses.get(i))) / (speciesList.size() - i);
        	for(int j = i + 1; j < speciesList.size(); j++) {
        		fitnesses.put(j, fitnesses.get(j) + adjusment);
        	}
        }
        
        // Carry over the champion if the species is big enough
        for(int i = 0; i < speciesList.size(); i++) {
        	if((speciesList.get(i).size() >= CHAMPION_THRESHOLD || i == 0) && amounts.get(i) >= 1) {
        		nets.add(new NEATNetwork(speciesList.get(i).get(0)));
        		amounts.put(i, amounts.get(i) - 1);
        	}
        }
        
        for(int i = 0; i < speciesList.size(); i++) {
        	 
        	if(amounts.get(i) == 0) {continue;}
        	
        	if(amounts.get(i) == 1) {
        		speciesList.get(i).get(0).mutate();
        		nets.add(new NEATNetwork(speciesList.get(i).get(0)));
        		continue;
        	}
        	
        	// Mate until the portion of offspring through mating is reached
        	long mateAmount = Math.round(amounts.get(i) * (1 - PORTION_MUTATION));
        	if(mateAmount != 0 && speciesList.get(i).size() > 1) {
        		amounts.put(i, amounts.get(i) - mateAmount);
        		while(mateAmount > 0) {
        			for(int j = 1; j < speciesList.get(i).size(); j++) {
        				for(int k = 0; k < j; k++) {
        					nets.add(speciesList.get(0).get(0).mate(speciesList.get(i).get(j), speciesList.get(i).get(k)));
        					mateAmount--;
        					if(mateAmount == 0) {break;}
        				}
        		
        				if(mateAmount == 0) {break;}
        			}
        		}
        	}
        	
        	// Add mutated copies to fill the rest of the species to the next 
        	// generation
        	while(amounts.get(i) > 0) {
        		for(int j = 0; j < speciesList.get(i).size(); j++) {
        			NEATNetwork mutated = new NEATNetwork(speciesList.get(i).get(j));
        			mutated.mutate();
        			nets.add(mutated);
        			amounts.put(i, amounts.get(i) - 1);
        			
        			if(amounts.get(i) == 0) {break;}
        		}
        	}
        }
        
        // Should for whatever reason not all slots been filled, like through
        // rounding errors, add new empty nets
        while(nets.size() < networkCount) {
        	nets.add(new NEATNetwork(inputCount, outputCount, this));
        }
    }
    
    /**
     * Print the current maximum fitness
     * @param high Highest previously achieved value
     */
    public double getMaxFitness()
    {
        double maxFitness = Double.NEGATIVE_INFINITY;
        
        for(NEATNetwork specimen : nets)
        {
            if( specimen.fitness > maxFitness)
            {
                maxFitness = specimen.fitness;
            }
        }
        
        return maxFitness;
    }
    
    /**
     * Tells a specific network to process the given inputs and return the 
     * result.
     * @param inputs Array of double values to be used in the input nodes.
     * @param index Index of the network in the "nets"-list
     * @return Returns an array the output value. 
     */
    public double[] processNetwork(double[] inputs,int index)
    {   
    	if(usesBias) {
    		double[] inputsBiased = new double[inputs.length + 1];
    		for(int i = 0; i < inputs.length; i++) {
    			inputsBiased[i] = inputs[i];
    		}
    		
    		inputsBiased[inputs.length] = 1.0;
    		return nets.get(index).process(inputsBiased);
    	}
    	
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
         * @return Returns whether the two integer pairs are equal
         */
        public boolean compare(IntegerPair p)
        {
            return (this.first == p.first && this.second == p.second);
        }
    }
}