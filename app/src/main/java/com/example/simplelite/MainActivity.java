package com.example.simplelite;

import android.app.Activity;
import androidx.appcompat.app.AppCompatActivity;
import android.content.res.AssetFileDescriptor;
import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Button;



import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.io.IOException;


public class MainActivity extends AppCompatActivity {

    private EditText Input;
    private TextView Output;
    Interpreter tflite;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Input=(EditText)findViewById(R.id.userID);
        Output=(TextView)findViewById(R.id.resultview);
        Button button =(Button)findViewById(R.id.result_button);



        try{
            tflite= new Interpreter(loadModelFile());
        }catch(Exception e){
            e.printStackTrace();
        }

        button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                float prediction=inference(Input.getText().toString());
                Output.setText(Float.toString(prediction));
            }

        });
    }

        public float inference(String s){
            float [] inputValue=new float[1];
            inputValue[0]=Float.valueOf(s);

            float[][] outputValue=new float[1][1];
            tflite.run(inputValue,outputValue);

            float inferredValue=outputValue[0][0];
            return inferredValue;
        }


    // 모델을 읽어오는 함수
    // MappedByteBuffer 바이트 버퍼를 Interpreter 객체에 전달하여 모델 해석
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("converted_model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


}
