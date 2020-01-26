/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Example;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import NEATLib.NEAT;

/**
 * Simple Implementation to test the NEAT-Implementation
 */
public class Main {
    /// Wanted amount of specimen per generation
    public static int networks = 100;
    /// Reference to the NEAT instance.
    public static NEAT neat;
    
    public static int h = 10;
    public static int v = 10;
    
    /**
     * Main function. Lets the neural nets play TicTacToe.
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // Make a new NEAT instande
        neat = new NEAT(h*v, h*v, networks, false);
        int generation = 0;
        
        // Main Loop
        while(true){
            System.out.println("Running Generation " + generation++);
            // Run the current generation
            newGeneration();
            // Print the results
            neat.printMaxFitness();
            // IDEA: Make a menu
            // Get User input before continuing
            try{System.in.read();} catch (IOException e){}
            // Let the NEAT advance to the next Generation
            neat.resetFitness();
            neat.advanceGeneration();
        }
    }
    
    /**
     * Test if the TicTacToe game is over and who won.
     * @param field Array present
     * @return returns the answer to the test (-1 Player B won, 0 no one won yet,
     * 1 Player A won, 2 draw)
    */
    public static int testArray(int[] field){
        // Checking horizontals
        int result;
        
        for(int i = 0; i < 9; i+=3){
            result = 0;
            
            for(int j = 0; j < 3; j++){
                result += field[i+j];
            }
            
            if(result == -3){
                return -1;
            } else if(result == 3){
                return 1;
            }
        }
        
        // Checking verticals
        for(int i = 0; i < 3; i++){
            result = 0;
            for(int j = 0; j < 9; j+=3){
                result += field[i+j];
            }
            
            if(result == -3){
                return -1;
            } else if(result == 3){
                return 1;
            }
        }
        result = 0;
        
        // Checking diagonals
        for (int i = 0; i < 9; i+=4){
            result += field[i];
        }
        
        if(result == -3){
            return -1;
        } else if (result == 3){
            return 1;
        }
        result = 0;
        
        for (int i = 2; i < 7; i+= 2 ){
            result += field[i];
        }
        
        if(result == -3){
            return -1;
        } else if(result == 3){
            return 1;
        }
        
        // Check if game is over
        for(int i : field){
            if(i == 0){
                return 0;
            }
        }
        
        return 2;
    }
    
    /**
     * Play TicTacToe with all nets. Everyone is once first and once second 
     * player
     */
    public static void runGeneration(){
        for (int a = 0; a < networks; a++){
            for(int b = 0; b < networks; b++){
                if(a==b){continue;}
                
                int field[] = {0,0,0,0,0,0,0,0,0};
                int result;
                
                // Play TicTacToe
                do {
                    // Send field to network A
                    double[] inputs = new double[9];
                    for(int k = 0; k < 9; k++){
                        inputs[k] = (double) field[k];
                    }
                    
                    double[] output = neat.processNetwork(inputs,a);
                    
                    // Apply highest value
                    double value = -1;
                    int index = 0;
                    for(int k = 0; k < 9; k++){    
                        if(field[k] == 0){
                            if(output[k] > value){
                                value = output[k];
                                index = k;
                            }
                        }
                    }
                    
                    field[index] = 1;
                    
                    result = testArray(field);
                    if(result != 0){break;}
                    
                    // Let network B process the field
                    for(int k = 0; k < 9; k++){
                        inputs[k] = (double) field[k];
                    }
                    
                    output = neat.processNetwork(inputs,b);
                    
                    // Apply highest value
                    value = -1;
                    index = 0;
                    for(int k = 0; k < 9; k++){    
                        if(field[k] == 0){
                            if(output[k] > value){
                                value = output[k];
                                index = k;
                            }
                        }
                    }
                    
                    field[index] = -1;
                    result = testArray(field);
                } while(result == 0);
                
                // Apply results
                switch (result) {
                    // Increase Fitness of network A by 2 if it won
                    case 1:
                        neat.addFitness(2,a);
                        break;
                    // Increase Fitness of Network B by 2 if it won
                    case -1:
                        neat.addFitness(2,b);
                        break;
                    // Increase Fitness of both by 1 if it was a draw
                    default:
                        neat.addFitness(1, a);
                        neat.addFitness(1, b);
                        break;
                }
            }
        }
    }

    public static void newGeneration(){
        for (int i = 0; i < networks; i++){
            for(int j = 0; j < networks; j++)
            {
                if(i == j){continue;}
                
                int[] field = new int[h*v];
                for(int p = 0; p < field.length; p++)
                {
                    field[p] = 0;
                }
                
                field[0] = 1;
                field[field.length - 1] = -1;
                
                int result = 0;
                
                do{
                // Let Network I play
                double[] inputs = new double[field.length];
                for(int p = 0; p < field.length; p++)
                {
                    inputs[p] = (double) field[p];
                }
                
                double[] output = neat.processNetwork(inputs,i);
                
                List<Integer> positions = new ArrayList<>();
                
                for(int p = 0; p < h*v; p++){
                    if(field[p] == 1){
                        for(int o = -1; o < 2; o += 2){
                            try{
                                if(field[p + o] == 0 && !positions.contains(p+o)){
                                    positions.add(p+o);
                                }
                            } catch (IndexOutOfBoundsException e){}
                            
                            try
                            {
                                if(field[p + v*o] == 0 && !positions.contains(p+v*o)){
                                    positions.add(p + v*o);
                                }
                            } catch (IndexOutOfBoundsException e){}
                        }
                    }
                }
                
                if(positions.isEmpty()){
                    result = -1;
                    break;
                }
                
                double value = -1;
                int index = 0;
                for(int p : positions){
                    if(output[p] > value){
                        value = output[p];
                        index = p;
                    }
                }
                
                field[index] = 1; 
                
                positions.clear();
                
                // Let Network J play
                for(int p = 0; p < field.length; p++)
                {
                    inputs[p] = (double) field[p];
                }
                
                output = neat.processNetwork(inputs,i);
                
                for(int p = 0; p < h*v; p++){
                    if(field[p] == -1){
                        for(int o = -1; o < 2; o += 2){
                            try{
                                if(field[p + o] == 0 && !positions.contains(p+o)){
                                    positions.add(p+o);
                                }
                            } catch (IndexOutOfBoundsException e){}
                            
                            try
                            {
                                if(field[p + v*o] == 0 && !positions.contains(p+v*o)){
                                    positions.add(p + v*o);
                                }
                            } catch (IndexOutOfBoundsException e){}
                        }
                    }
                }
                
                if(positions.isEmpty()){
                    result = 1;
                    break;
                }
                
                value = -1;
                index = 0;
                for(int p : positions){
                    if(output[p] > value){
                        value = output[p];
                        index = p;
                    }
                }
                
                field[index] = -1; 
                
                positions.clear();
                
                }while(true);
                
                
                // Apply results
                switch (result) {
                    // Increase Fitness of network A by 2 if it won
                    case 1:
                        neat.addFitness(2,i);
                        break;
                    // Increase Fitness of Network B by 2 if it won
                    case -1:
                        neat.addFitness(2,j);
                        break;
                }
            }
        }
    }
}