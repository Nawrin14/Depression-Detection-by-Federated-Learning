package com.example.client;

import android.app.AlertDialog;
import android.app.Dialog;
import android.content.Context;
import android.content.DialogInterface;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.DialogFragment;

public class GenderChoiceDialog extends DialogFragment {

    int position = 0;

    public interface GenderChoiceListener{
        void onPositiveButtonClicked(String[] list, int position);
    }

    GenderChoiceListener mListener;

    @Override
    public void onAttach(Context context) {
        super.onAttach(context);
        try{
            mListener = (GenderChoiceListener) context;
        }
        catch(Exception e){
            throw new ClassCastException(getActivity().toString() + "GenderChoiceListener must be implemented!");
        }
    }

    @NonNull
    @Override
    public Dialog onCreateDialog(@Nullable Bundle savedInstanceState) {

        AlertDialog.Builder builder = new AlertDialog.Builder(getActivity());
        final String[] list = getActivity().getResources().getStringArray(R.array.list_gender);
        builder.setTitle("Select your gender").setSingleChoiceItems(list, position, new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialogInterface, int i) {
                position = i;
            }
        })
        .setPositiveButton("Submit", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialogInterface, int i) {
                mListener.onPositiveButtonClicked(list, position);
            }
        });

        return builder.create();
    }
}