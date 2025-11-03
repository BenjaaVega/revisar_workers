import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from transbank.webpay.webpay_plus.transaction import Transaction
from transbank.common.options import WebpayOptions
from transbank.common.integration_type import IntegrationType

class WebPayService:
    def __init__(self):
        # Configuración para ambiente de integración (no producción)
        self.commerce_code = os.getenv("WEBPAY_COMMERCE_CODE", "597055555532")
        self.api_key = os.getenv("WEBPAY_API_KEY", "579B532A7440BB0C9079DED94D31EA1615BACEB56610332264630D42D0A36B1C")
        
        # Crear opciones para la nueva versión del SDK
        self.options = WebpayOptions(
            commerce_code=self.commerce_code,
            api_key=self.api_key,
            integration_type=IntegrationType.TEST
        )
    
    def create_transaction(self, amount: float, order_id: str, session_id: str, 
                          return_url: str) -> Dict[str, Any]:
        """
        Crear una transacción de WebPay Plus
        """
        try:
            # Crear instancia de Transaction con options
            transaction = Transaction(self.options)
            response = transaction.create(
                buy_order=order_id,
                session_id=session_id,
                amount=amount,
                return_url=return_url
            )
            
            return {
                "success": True,
                "token": response["token"],
                "url": response["url"],
                "amount": amount,
                "order_id": order_id
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def commit_transaction(self, token: str) -> Dict[str, Any]:
        """
        Confirmar una transacción
        """
        try:
            transaction = Transaction(self.options)
            response = transaction.commit(token)
            
            return {
                "success": True,
                "transaction": {
                    "vci": response["vci"],
                    "amount": response["amount"],
                    "status": response["status"],
                    "buy_order": response["buy_order"],
                    "session_id": response["session_id"],
                    "card_detail": response["card_detail"],
                    "accounting_date": response["accounting_date"],
                    "transaction_date": response["transaction_date"],
                    "authorization_code": response["authorization_code"],
                    "payment_type_code": response["payment_type_code"],
                    "response_code": response["response_code"],
                    "installments_number": response["installments_number"]
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_transaction_status(self, token: str) -> Dict[str, Any]:
        """
        Obtener estado de una transacción
        """
        try:
            transaction = Transaction(self.options)
            response = transaction.status(token)
            
            return {
                "success": True,
                "transaction": {
                    "vci": response["vci"],
                    "amount": response["amount"],
                    "status": response["status"],
                    "buy_order": response["buy_order"],
                    "session_id": response["session_id"],
                    "card_detail": response["card_detail"],
                    "accounting_date": response["accounting_date"],
                    "transaction_date": response["transaction_date"],
                    "authorization_code": response["authorization_code"],
                    "payment_type_code": response["payment_type_code"],
                    "response_code": response["response_code"],
                    "installments_number": response["installments_number"]
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }