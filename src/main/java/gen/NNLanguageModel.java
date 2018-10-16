package gen;

import java.io.*;
import java.net.*;

import main.PathList;

public class NNLanguageModel {
    PrintWriter out;
    BufferedReader in;
    Socket socket = null;
    ServerSocket serverSocket = null;

    public NNLanguageModel(String modelName, String datasetName) {
        try {
            serverSocket = new ServerSocket(32000);
            String command = "python " + PathList.NN_LANGUAGE_MODEL_PATH;
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

    public double scoreSentence(String sent) {
        try {
            send(sent);
            flush();
            return Double.parseDouble(recv());
        } catch (Exception e) {
            System.exit(1);
        }
        return 0.0;
    }

    private void send(String msg) {
        out.println(msg);
    }

    private void flush() {
        out.flush();
    }

    private String recv() throws Exception {
        return in.readLine();
    }
}
