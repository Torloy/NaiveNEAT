package NEATLib;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * This class represents a single NEAT-Network
 */
public class NEATNetwork
{
    /// Reference to the NEAT instance
    NEAT reference;
    
    /// List of node-genes in this network
    List<Node> nodes = new ArrayList<>();
    /// List of connection-genes in this network
    List<Connection> connections = new ArrayList<>();
    
    /// Number of output nodes in this network
    int outputs = 0;
    /// Number of input nodes in this network
    int inputs = 0;
    /// Fitness of this network
    double fitness = 0.0;
    
    // *STRUCTORS --------------------------------------------------------------
    
    /**
     * Constructor for a new NEAT neural network. This one is used in the
     * initialization process of a new run
     * @param inNodes Amount of wanted input nodes.
     * @param outNodes Amount of wanted output nodes.
     * @param ref Reference to the neat network collection.
     */
    public NEATNetwork(int inNodes, int outNodes, NEAT ref)
    {
        // Write to members
        outputs = outNodes;
        inputs = inNodes;
        reference = ref;
        
        // Add input nodes
        for(int i = 0; i < inNodes; i++)
        {
            nodes.add(new Node(Type.INPUT,null,null,i));
        }
        
        // Add output nodes and the possible connections towards the input nodes
        for(int i = 0; i < outNodes; i++)
        {
            Node n = new Node(Type.OUTPUT,null,null,i + inNodes);
            
            // Add the input nodes as possible connections
            for(int j = 0; j < inNodes;j++)
            {
                n.possibleConnections.add(nodes.get(j));
            }
            
            nodes.add(n);
        }
        
        // Add all possible connections towards the output nodes
        for(int i = 0; i < inNodes; i++)
        {
            for(int j = inNodes; j < inNodes + outNodes; j++)
            {
                nodes.get(i).possibleConnections.add(nodes.get(j));
            }
        }
        
        // Mutate the network for the first time
        mutate();
    }
    
    /**
     * Constructor for a new NEAT network. Used during mating process.
     * @param inNodes Amount of input nodes.
     * @param outNodes Amount of output nodes.
     * @param nodes List of node genes in this network.
     * @param connections List of connection genes in this network.
     * @param ref Reference to the NEAT instance
     */
    private NEATNetwork(
            int inNodes, 
            int outNodes, 
            List<Node> nodes, 
            List<Connection> connections, 
            NEAT ref)
    {
        inputs = inNodes;
        outputs = outNodes;
        this.nodes = nodes;
        this.connections = connections;
        reference = ref;
    }
    
    // METHODS -----------------------------------------------------------------
    
    /**
     * Increases the fitness
     * @param value Value with which the fitness is increased
     */
    public void addFitness(double value)
    {
        fitness += value;
    }
    
    /**
     * Calculate the delta between two nets.
     * @param b Net to be compared with.
     * @param weightC1 Weight on the excess genes
     * @param weightC2 Weight on the dijoint genes
     * @param weightC3 Weight on the average difference in matching weights
     * @return Returns the delta between the two nets
     */
    public double compareCompatibility(
            NEATNetwork b, 
            double weightC1, 
            double weightC2, 
            double weightC3)
    {
        // Number of connection genes in the bigger specimen
        int n = connections.size() > b.connections.size() ? connections.size() : b.connections.size();
        /* Number of excess genes with innovation numbers bigger than the 
         * highest of the other genome */
        int excess = 0;
        /* Number of disjoint genes with innovation numbers not existant in the
         * other genome and not pertaining to the excess genes */
        int disjoint = 0;
        // Number of matching genes
        int matching = 0;
        // Difference in weight between the matching genes
        double weightDifference = 0;
        
        // Sort the connection list after the innnovation number
        Collections.sort(connections, (Connection c1, Connection c2) -> {
            return (c1.innovationNumber - c2.innovationNumber) * -1;
        });
        
        Collections.sort(b.connections, (Connection c1, Connection c2) -> {
            return (c1.innovationNumber - c2.innovationNumber) * -1;
        });
        
        // Calculate the different values
        int index = 0;
        for(Connection c : connections)
        {
            // If the index is at the end of b but not c add excess genes
            if(index == b.connections.size())
            {
                excess++;
            } 
            /* if the innovation numbers match increase the matching numbers and
             * add the absolute weight difference */
            else if(c.innovationNumber == 
                    b.connections.get(index).innovationNumber)
            {
                matching++;
                weightDifference += Math.abs(c.weight - b.connections.get(index).weight);
                index++;
            } 
            // if the innovation number is smaller add a disjoint gene    
            else if(c.innovationNumber < 
                    b.connections.get(index).innovationNumber)
            {
                disjoint++;
            } 
            /* if the innovation number is bigger store the amount of genes in a
             * temp val and decide after if they are disjoint or excess genes */
            else {
                int temp = 0;
                
                while(c.innovationNumber > b.connections.get(index).innovationNumber){
                    temp++;
                    index++;
                    if(index == b.connections.size()){
                        excess += temp;
                        break;
                    } else if(c.innovationNumber <= b.connections.get(index).innovationNumber){
                        disjoint += temp;
                        break;
                    }
                }
            }
        }

        // Average matching gene's weight differences
        weightDifference /= matching;
        
        // Calculate delta;
        return (weightC1 * excess / n) 
                + (weightC2 * disjoint / n) 
                + (weightC3 * weightDifference);
    }
    
    /**
     * Mates two networks and returns a new network
     * @param a Parent A for the new network
     * @param b Parent B for the new network
     * @return Returns the offspring of A and B as a new network
     */
    public NEATNetwork mate(NEATNetwork a, NEATNetwork b)
    {
        /// Offspring to be returned
        NEATNetwork offspring;
        /// The fitter parent network
        NEATNetwork better = a.fitness > b.fitness ? a : b;
        /// The less fit parent network
        NEATNetwork worse = better == a ? b : a;
        /// Map of the nodes for the offspring ordered by node ID
        Map<Integer,Node> newNodeMap = new TreeMap<>();
        /// List of the connections for the offspring
        List<Connection> newConnections = new ArrayList<>();
        /// List of kept connections from the parents
        Map<Integer,Connection> keptConnections = new TreeMap<>();
        /// List of new nodes for the offspring
        List<Node> newNodes = new ArrayList<>();
        
        // Proper sorting
        Collections.sort(
                better.connections, 
                (Connection a1, Connection b1) -> 
                        a1.innovationNumber - b1.innovationNumber);
        Collections.sort(
                worse.connections, 
                (Connection a1, Connection b1) -> 
                        a1.innovationNumber - b1.innovationNumber);
        
        // Mark all better conections as to be transferred
        for(Connection c : better.connections){
            keptConnections.put(c.innovationNumber,c);
        }
        
        // Get a portion of the matching genes in relation to their fitness
        // IDEA: Use Seed
        for(Connection c : worse.connections){
            if(keptConnections.containsKey(c.innovationNumber) 
                    && Math.random() < worse.fitness / (better.fitness + worse.fitness)){
                keptConnections.replace(c.innovationNumber, c);
            }
        }
        
        // Apply to new list
        for(Node n: better.nodes){
            newNodeMap.put(n.nodeID,new Node(n));
        }
        
        // Apply possible connections
        for(Node n : newNodeMap.values()){
            for(Node nc : newNodeMap.values()){
                if(n == nc || n.t == nc.t){continue;}
                
                n.possibleConnections.add(nc);
            }
        }
        
        // Apply new connections
        for(Connection c : keptConnections.values()){
            newConnections.add(new Connection(
                    newNodeMap.get(c.inRef.nodeID),
                    newNodeMap.get(c.outRef.nodeID),
                    c.enabled
            ));
        }
        
        // Put nodes into a normal array list
        for(Node n : newNodeMap.values()){
            newNodes.add(n);
        }
        
        Collections.sort(newNodes, (Node na, Node nb) -> nb.nodeID - na.nodeID);
        
        // Apply to new network
        offspring = new NEATNetwork(inputs, outputs, newNodes, newConnections, a.reference);
        
        return offspring;
    }
    
    // IDEA: Adjust mutation function to mirror the paper
    /**
     * Mutate a network with given propabilities
     */
    public void mutate()
    {
        // Test if a structural change is to be made
        if(Math.random() < reference.PROP_STRUCTURE || connections.isEmpty())
        {
            // Test which structural mutation is applied
            if(Math.random() < reference.PROP_CONNECTION 
                    || connections.isEmpty())
            {
                // Add a new connection
                Node n1 = nodes.get((int) (Math.random() * nodes.size()));
                Node n2 = n1.getPossibility();
                if(n2 == null) {return;}
                
                if(n1.t == Type.OUTPUT || n2.layerNumber < n1.layerNumber)
                {
                    connections.add(new Connection(n2,n1,true));
                    return;
                }
                
                connections.add(new Connection(n1,n2,true));
                return;
            } 
            
            // Get the original connection and disable it
            Connection original = 
                    connections.get((int) (Math.random() * connections.size()));
            original.enabled = false;
            
            // Add a new node
            Node insertNode = new Node(
                    Type.HIDDEN,
                    original.inRef,
                    original.outRef,
                    0);
            nodes.add(insertNode);
            insertNode.nodeID = nodes.indexOf(insertNode);
            
            // Add a new connection towards the new node
            connections.add(new Connection(insertNode, original.outRef, true));
            
            // Add a new connection with weight 1 from the new node
            Connection newIn = new Connection(original.inRef, insertNode, true);
            newIn.weight = 1;
            connections.add(newIn);
        } 
        else 
        {
            // Test wether to toggle a connection or weight
            if(Math.random() < reference.PROP_TOGGLE)
            {
                connections.get((int) (Math.random() * connections.size())).toggle();
                return;
            }
            connections.get((int) (Math.random() * connections.size())).adjustWeight();
        }
    }
    
    /**
     * Processes the net with a given input
     * @param inputs Array of input values
     * @return Returns an array corresponding to the calculated outputs
     */
    public double[] process(double[] inputs)
    {
        Collections.sort(connections, (Connection a, Connection b) -> 
                (int) (b.inRef.layerNumber - a.inRef.layerNumber)
        );
        
        // Assign the input values
        int index = 0;
        for(Node n: nodes)
        {
            if(n.t == Type.INPUT)
            {
                n.currentValue = inputs[index];
                index++;
            }
            else
            {
                n.currentValue = 0;
            }
        }
        
        // Go through the connections and assign the values
        for(Connection c : connections)
        {
            if(c.inRef.t != Type.INPUT)
            {
                c.inRef.activate();
            }
            c.outRef.currentValue += c.inRef.currentValue * c.weight;
        }
        
        // Get the values in the output
        double[] output = new double[outputs];
        index = 0;
        for(Node n: nodes)
        {
            if(n.t == Type.OUTPUT)
            {
                n.activate();
                output[index] = n.currentValue;
                index++;
            }
        }
        
        return output;
    }
    
    // INTERNAL CLASSES --------------------------------------------------------
    
    /**
     * Internal class representing a connection gene in the genome
     */
    class Connection
    {
        /// Input node to the connection
        Node inRef;
        /// Output node to the connection
        Node outRef; 
        
        /// Indicates wether or not the connection is active
        boolean enabled = true;
        /// The weight f the connection
        double weight = 1;
        /// The innovation number of the connection
        int innovationNumber;
        
        /**
         * Creates a new Connection between Nodes
         * @param input Input node. Never assign an output to it.
         * @param output Output node. Never assign an input to it.
         */
        public Connection(Node input, Node output, boolean enabled)
        {
            // Check wether the connection is the right way around
            input.checkType(Type.OUTPUT);
            output.checkType(Type.INPUT);
            
            // Set the proper references
            inRef = input;
            outRef = output;
            this.enabled = enabled;
            
            // Remove the connection
            inRef.removePossibility(output);
            outRef.removePossibility(input);
            
            // Set the innovation number
            innovationNumber = reference.addInnovation(
                    nodes.indexOf(inRef), 
                    nodes.indexOf(outRef)
            );
            
            // Set a random weight
            adjustWeight();
        }
        
        // IDEA: Add config value for weight range
        /**
         * Randomly adjust the weight
         */
        public void adjustWeight()
        {
            weight += Math.random() * 2 - 1;
        }
        
        /**
         * Toggles the connection
         */
        public void toggle()
        {
            enabled = !enabled;
        }
    }
    
    /**
     * Internal class representing a node in the network
     */
    class Node
    {
        /// Possible connections to be made with other nodes
        List<Node> possibleConnections = new ArrayList<>();
        /// Type of the node
        Type t;
        
        /// Layer number to keep track of the sequence 
        double layerNumber;
        /// Current value of the processing 
        double currentValue = 0;
        /// ID of the node
        int nodeID = 0;
        
        /**
         * Creates a new Node
         * @param type Defines the type of the node
         * @param input Reference to the input node of the split connection
         * @param output Reference to the output node of the split connection
         */
        public Node(Type type, Node input, Node output, int ID)
        {
            t = type;
            nodeID = ID;
            
            switch(type){
                case INPUT:
                    layerNumber = Double.MIN_VALUE;
                    break;
                case OUTPUT:
                    layerNumber = Double.MAX_VALUE;
                    break;
                case HIDDEN:
                    for(Node n : nodes){
                        possibleConnections.add(n);
                    }
                
                    possibleConnections.remove(input);
                    possibleConnections.remove(output);
                    
                    layerNumber = input.layerNumber * 0.5  + output.layerNumber * 0.5;
        
                    break;
            }
        }
        
        /**
         * Constructor for a node using another node
         * @param old Node to be replicated
         */
        public Node(Node old)
        {
            t = old.t;
            nodeID = old.nodeID;
            layerNumber = old.layerNumber;
        }
        
        // IDEA: Allow for different actiovation functions
        /**
         * Activates the node using a sigmoid function on its currentValue.
         */
        public void activate()
        {
            currentValue = 1.0 / (1.0 + Math.pow(Math.E,-currentValue));
        }
        
        /**
         * Checks wether a node has been passed as an illegal argument
         * @param t Type of the node that has been passed as an illegal argument
         */
        public void checkType(Type t)
        {
            if(t == this.t){
                throw new IllegalArgumentException();
            }
        }
        
        /**
         * Remove one of the possible connections.
         * @param n Node to be exempt from the list
         */
        public void removePossibility(Node n)
        {
            possibleConnections.remove(n);
        }
        
        // IDEA: Use seed
        /**
         * Get a node with which a connection can be established 
         * @return Return the possible node
         */
        public Node getPossibility()
        {
            if(possibleConnections.isEmpty()){return null;}
            
            return possibleConnections.get((int) (Math.random() * possibleConnections.size()));
        }
    }
    
    /**
     * Enum holding the types of nodes in the network
     */
    enum Type {INPUT,HIDDEN,OUTPUT};
}