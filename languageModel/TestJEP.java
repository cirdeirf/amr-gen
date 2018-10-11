import jep.Jep;
import jep.JepException;

public class TestJEP {
    public static void main(String args[]) {
        try (Jep jep = new Jep()) {
            jep.eval("import sys, os");
            jep.eval("sys.path.append(os.getcwd())");

            jep.eval("from lmTest import NNLanguageModel");
            jep.eval("lm = NNLanguageModel('awd_lstm_lm_600', 'wikitext-2')");

            String sent0 =
                "I that that this is a disgraceful situation that should not be tolerated";
            String sent1 =
                "I submit that this is a disgraceful situation that should not be tolerated";

            jep.set("sent", sent0);
            jep.eval("score = lm.score(sent)");
            Object score0 = jep.getValue("score");
            jep.set("sent", sent1);
            jep.eval("score = lm.score(sent)");
            Object score1 = jep.getValue("score");

            System.out.println(score0);
            System.out.println(score1);
            System.out.println((double) score0 > (double) score1);
        } catch (JepException e) {
            System.out.println("An error occurred: " + e.getMessage());
        }
    }
}
