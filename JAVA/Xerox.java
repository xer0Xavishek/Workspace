
import java.util.Scanner;
public class Xerox{
    public static void main(String[] args) {
        System.out.print("Enter your name : ");
        Scanner input = new Scanner(System.in);
        String name= input.nextLine();
        System.out.println("Your name is "+ name);
        System.out.print("Enter your age : ");
        int age=input.nextInt();
        System.out.println("Your age is :"+age);
    }
}