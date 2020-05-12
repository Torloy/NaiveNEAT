package Example;

import tasks.XORTest;

/**
 * Simple Implementation to test the NEAT-Implementation
 */
public class Main {
    /**
     * Main function. Lets the neural nets process different tasks.
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        XORTest t = new XORTest();
        t.runTest();
    }
}