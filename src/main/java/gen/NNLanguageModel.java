package gen;

import main.PathList;

import java.io.*;
import java.net.*;
import java.util.*;

/**
 * This class provides the means to communicate with a python script that
 * scores natural language sentence using a neural network language model (with
 * the help of the GluonNLP toolkit).
 */

public class NNLanguageModel {
    PrintWriter out;
    BufferedReader in;
    Socket socket = null;
    ServerSocket serverSocket = null;

    /**
     * Create a new connection to a script providing a neural network language
     * model.
     * @param modelName the type of neural network language model to use
     * (possible values: "awd_lstm_lm_1150", "awd_lstm_lm_600",
     * "standard_lstm_lm_1500", "standard_lstm_lm_650", "standard_lstm_lm_200")
     */
    public NNLanguageModel(String modelName) {
        try {
            serverSocket = new ServerSocket(32000);
            String command = "python " + PathList.NN_LANGUAGE_MODEL_PATH
                + " -m " + modelName;
            Runtime.getRuntime().exec(command);
            socket = serverSocket.accept();
            System.out.println("Connected");
            out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(
                                      socket.getOutputStream())),
                true);
            in = new BufferedReader(
                new InputStreamReader(socket.getInputStream()));
        } catch (Exception e) {
            System.exit(1);
        }
    }

    /**
     * Send a list of sentences for scoring to a python script. The sentences
     * will be prepared beforehand: Each sentence gets tokens indicating the
     * beginning and end of a sentence and shorter sentences get a padding after
     * the end token. Additionally, the amount of sentences is prepended to the
     * final string that is sent to the python script determining the batch
     * size.
     * @param sentences a list of strings (prediction values) that is to be
     * scored
     * @return the list of scores given by the neural network language model
     * (negative logarithmic probabilities)
     */
    public List<Double> scoreSentences(ArrayList<String> sentences) {
        // calculate the length of the longest sentence
        int maxLength = 0;
        for (String s : sentences) {
            if (maxLength < s.split(" ").length) {
                maxLength = s.split(" ").length;
            }
        }
        // prepare the string that is to be sent to the script
        int batchSize = sentences.size();
        // preprend amount of sentences / batch size
        String lineSend = Integer.toString(batchSize);
        for (String s : sentences) {
            // indicate beginning and end of sentence
            lineSend += " <bos> " + s + " <eos>";
            // fill up shorter sentences such that all sentences may be
            // processed in one call
            for (int i = 0; i < (maxLength - s.split(" ").length); i++) {
                lineSend += " <pad>";
            }
        }
        try {
            send(lineSend);
            flush();
            String lineRecv = recv();
            List<Double> scores = new ArrayList<>();
            // split up the received string (representing the scores)
            for (String s : Arrays.asList(lineRecv.split(" "))) {
                scores.add(Double.parseDouble(s));
            }
            return scores;
        } catch (Exception e) {
            System.exit(1);
        }
        return null;
    }

    /**
     * Helper function that writes a message to the python script.
     * @param msg the string that is to be sent
     */
    private void send(String msg) {
        out.println(msg);
    }

    /**
     * Helper function that flushes the message, i.e., actually sends it out.
     */
    private void flush() {
        out.flush();
    }

    /**
     * Helper function for receiving any incoming answer from the python script.
     */
    private String recv() throws Exception {
        return in.readLine();
    }
}
