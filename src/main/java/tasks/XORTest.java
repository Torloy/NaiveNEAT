package tasks;

import NEATLib.NEAT;

public class XORTest {
	
	public NEAT neat;
	public int networks = 1000;
	
	public int runTest() {
		neat = new NEAT(2, 1, networks);
		int generation = 0;
		
		while(true) {
			System.out.println("Running Generation " + generation++);
            runGeneration();
            
            System.out.print("	Maximum fitness: ");
            System.out.println(neat.getMaxFitness());
            
            if(neat.getMaxFitness() > 95) {
            	break;
            }
            
            System.out.println("	Advancing Generation");
            neat.advanceGeneration();
            
            System.out.println("	Resetting Fitness.");
            neat.resetFitness();
            
            System.out.println("	Finished Generation");
		}
		
		return 0;
	}
	
	void runGeneration() {
		for(int i = 0; i < networks; i++) {
			for(int a = 0; a <= 1; a++) {
				for(int b = 0; b <= 1; b++) {
					double[] inputs = {(double) a, (double) b};
					
					double[] output = neat.processNetwork(inputs, i);
					
					if((a ^ b) == 0) {
						neat.addFitness(25 * (1 - output[0]), i);
					} else {
						neat.addFitness(25 * output[0], i);
					}
				}
			}
		}
	}
}