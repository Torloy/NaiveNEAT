/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
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