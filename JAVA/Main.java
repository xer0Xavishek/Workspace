    import java.util.Scanner;
    public class Main{
        public static void main(String[] args) {
            Scanner sc = new Scanner(System.in);
            System.out.println("Enter the string you want to repeat: ");
            String str = sc.nextLine();
            
            
            System.out.print("Enter the number of times you want to repeat the string: ");
            int n= sc.nextInt();

            String repeated = str.repeat(n);
            System.out.println(repeated);
        }
    }