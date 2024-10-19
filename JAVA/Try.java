import java.util.Scanner;


public class Try {
    public static void main(String[] args) {
        System.out.println("Enter your balance: ");
        Scanner input = new Scanner(System.in);
        int taka = input.nextInt();
        System.out.println("Your balance is " + taka);

        Scanner input2 = new Scanner(System.in);
        String name= input2.nextLine();
        System.out.println("Your name is " + name);
    }
}
